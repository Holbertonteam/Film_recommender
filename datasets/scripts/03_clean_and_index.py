"""Clean the raw + TMDB-enriched data and map raw user/movie ids to
contiguous 0-based indices for embedding layers.

Outputs (all in datasets/processed/):
  ratings_clean.parquet   user_idx, movie_idx, rating, timestamp
  movies.parquet          movie_idx, title, genres, poster_url
  users.parquet           user_idx, gender, age, occupation, zip_code  (bonus, not required by the core schema)
  user_mapping.parquet    user_id, user_idx    (reference, for tracing back to raw ids)
  movie_mapping.parquet   movie_id, movie_idx  (reference, for tracing back to raw ids)
"""

import pandas as pd

from config import MOVIES_ENRICHED_PATH, PROCESSED_DIR
from movielens_io import load_ratings, load_users


def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    n_before = len(ratings)

    ratings = ratings.dropna(subset=["user_id", "movie_id", "rating", "timestamp"])
    ratings = ratings.drop_duplicates(subset=["user_id", "movie_id", "timestamp"])
    ratings = ratings[ratings["rating"].between(1, 5)]

    n_after = len(ratings)
    if n_after != n_before:
        print(f"Dropped {n_before - n_after:,} invalid/duplicate rating rows.")
    return ratings


def build_index_mapping(ids: pd.Series, id_col: str, idx_col: str) -> pd.DataFrame:
    unique_ids = sorted(ids.unique())
    return pd.DataFrame({id_col: unique_ids, idx_col: range(len(unique_ids))})


def main() -> None:
    ratings = clean_ratings(load_ratings())
    users = load_users()
    movies = pd.read_csv(MOVIES_ENRICHED_PATH)

    user_mapping = build_index_mapping(ratings["user_id"], "user_id", "user_idx")
    movie_mapping = build_index_mapping(ratings["movie_id"], "movie_id", "movie_idx")

    ratings = ratings.merge(user_mapping, on="user_id").merge(
        movie_mapping, on="movie_id"
    )
    ratings_clean = ratings[["user_idx", "movie_idx", "rating", "timestamp"]].sort_values(
        ["user_idx", "timestamp"]
    )

    movies = movies.merge(movie_mapping, on="movie_id", how="inner")
    movies_out = movies[["movie_idx", "title", "genres", "poster_url"]].sort_values(
        "movie_idx"
    )

    users = users.merge(user_mapping, on="user_id", how="inner")
    users_out = users[
        ["user_idx", "gender", "age", "occupation", "zip_code"]
    ].sort_values("user_idx")

    ratings_clean.to_parquet(PROCESSED_DIR / "ratings_clean.parquet", index=False)
    movies_out.to_parquet(PROCESSED_DIR / "movies.parquet", index=False)
    users_out.to_parquet(PROCESSED_DIR / "users.parquet", index=False)
    user_mapping.to_parquet(PROCESSED_DIR / "user_mapping.parquet", index=False)
    movie_mapping.to_parquet(PROCESSED_DIR / "movie_mapping.parquet", index=False)

    print(f"\n{len(ratings_clean):,} ratings, {len(user_mapping):,} users, "
          f"{len(movie_mapping):,} movies -> {PROCESSED_DIR}")
    missing_posters = movies_out["poster_url"].isna().sum()
    if missing_posters:
        print(f"Note: {missing_posters:,} movies have no poster_url (TMDB match failed).")


if __name__ == "__main__":
    main()
