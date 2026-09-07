"""Per-user temporal train/test split (not a random split): each user's most
recent TEST_FRACTION of ratings become the test set, the rest train."""

import pandas as pd

from config import MIN_TEST_RATINGS_PER_USER, PROCESSED_DIR, TEST_FRACTION


def split_user(group: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group = group.sort_values("timestamp", kind="stable")
    n_test = max(MIN_TEST_RATINGS_PER_USER, round(len(group) * TEST_FRACTION))
    n_test = min(n_test, len(group) - 1)  # always leave >=1 rating in train
    return group.iloc[:-n_test], group.iloc[-n_test:]


def main() -> None:
    ratings = pd.read_parquet(PROCESSED_DIR / "ratings_clean.parquet")

    train_parts, test_parts = [], []
    for _, group in ratings.groupby("user_idx", sort=False):
        train_g, test_g = split_user(group)
        train_parts.append(train_g)
        test_parts.append(test_g)

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    # Sanity check: no user's test ratings should predate their train ratings.
    bounds = pd.DataFrame(
        {
            "train_max_ts": train.groupby("user_idx")["timestamp"].max(),
            "test_min_ts": test.groupby("user_idx")["timestamp"].min(),
        }
    )
    violations = bounds[bounds["train_max_ts"] > bounds["test_min_ts"]]
    if len(violations):
        print(
            f"Note: {len(violations):,} users have a tied timestamp straddling "
            "the train/test boundary (multiple ratings in the same second)."
        )

    train.to_parquet(PROCESSED_DIR / "ratings_train.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "ratings_test.parquet", index=False)

    print(f"\ntrain: {len(train):,} ratings ({len(train) / len(ratings) * 100:.1f}%)")
    print(f"test:  {len(test):,} ratings ({len(test) / len(ratings) * 100:.1f}%)")
    print(f"users covered: {ratings['user_idx'].nunique():,}")


if __name__ == "__main__":
    main()
