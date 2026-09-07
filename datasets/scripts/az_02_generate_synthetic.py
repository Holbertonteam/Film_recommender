"""Generate synthetic users + ratings on top of the real film catalog from
az_01_fetch_movies.py. Outputs go to processed_az/, same schema as processed/."""

import time

import numpy as np
import pandas as pd

from config import (
    AZ_MAX_RATINGS_PER_USER,
    AZ_MIN_RATINGS_PER_USER,
    AZ_MOVIES_RAW_PATH,
    AZ_N_SYNTHETIC_USERS,
    AZ_PROCESSED_DIR,
    AZ_RANDOM_SEED,
    AZ_TOP_N_MOVIES,
)

OCCUPATIONS = [
    "student",
    "teacher",
    "engineer",
    "artist",
    "doctor",
    "civil_servant",
    "retail",
    "unemployed",
    "self_employed",
    "retired",
]
SECONDS_PER_DAY = 86_400


def load_and_filter_movies() -> pd.DataFrame:
    movies = pd.read_csv(AZ_MOVIES_RAW_PATH)
    n_before = len(movies)
    movies = movies[movies["genres"].notna() & (movies["genres"] != "")]
    n_after = len(movies)
    if n_after != n_before:
        print(f"Dropped {n_before - n_after:,} films with no genre data (can't taste-match them).")

    movies = movies.assign(
        has_poster=movies["poster_url"].notna(),
        has_overview=movies["overview"].fillna("") != "",
    ).sort_values(
        ["vote_count", "vote_average", "has_poster", "has_overview"],
        ascending=False,
    )
    n_before = len(movies)
    movies = movies.head(AZ_TOP_N_MOVIES)
    print(f"Kept top {len(movies):,} of {n_before:,} films (by TMDB votes/rating).")
    return movies.drop(columns=["has_poster", "has_overview"]).reset_index(drop=True)


def build_genre_matrix(movies: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    genre_lists = movies["genres"].str.split("|")
    all_genres = sorted({g for genres in genre_lists for g in genres})
    genre_idx = {g: i for i, g in enumerate(all_genres)}

    matrix = np.zeros((len(movies), len(all_genres)), dtype=np.float32)
    for row, genres in enumerate(genre_lists):
        for g in genres:
            matrix[row, genre_idx[g]] = 1.0
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix = matrix / np.clip(row_sums, 1, None)
    return matrix, all_genres


def generate_users(rng: np.random.Generator, n_genres: int) -> pd.DataFrame:
    taste_vectors = rng.dirichlet(np.ones(n_genres), size=AZ_N_SYNTHETIC_USERS)
    n_ratings = rng.integers(
        AZ_MIN_RATINGS_PER_USER, AZ_MAX_RATINGS_PER_USER + 1, size=AZ_N_SYNTHETIC_USERS
    )
    rating_bias = rng.normal(0, 0.4, size=AZ_N_SYNTHETIC_USERS)
    gender = rng.choice(["M", "F"], size=AZ_N_SYNTHETIC_USERS)
    age = rng.integers(15, 70, size=AZ_N_SYNTHETIC_USERS)
    occupation = rng.choice(OCCUPATIONS, size=AZ_N_SYNTHETIC_USERS)
    zip_code = rng.integers(1000, 1200, size=AZ_N_SYNTHETIC_USERS)

    return pd.DataFrame(
        {
            "user_idx": np.arange(AZ_N_SYNTHETIC_USERS),
            "taste_vector": list(taste_vectors),
            "n_ratings": n_ratings,
            "rating_bias": rating_bias,
            "gender": gender,
            "age": age,
            "occupation": occupation,
            "zip_code": zip_code,
        }
    )


def generate_ratings(
    rng: np.random.Generator, users: pd.DataFrame, genre_matrix: np.ndarray
) -> pd.DataFrame:
    n_movies = genre_matrix.shape[0]
    now = int(time.time())
    parts = []

    for user in users.itertuples():
        affinity = genre_matrix @ user.taste_vector
        weights = np.exp(affinity * 4 - affinity.max() * 4)
        weights /= weights.sum()

        k = min(user.n_ratings, n_movies)
        movie_idx = rng.choice(n_movies, size=k, replace=False, p=weights)

        watched_affinity = affinity[movie_idx]
        z = (watched_affinity - watched_affinity.mean()) / (watched_affinity.std() + 1e-6)
        raw_score = 3.8 + z * 0.9 + user.rating_bias
        noisy_score = raw_score + rng.normal(0, 0.4, size=k)
        rating = np.clip(np.round(noisy_score), 1, 5).astype(np.int64)

        span_days = 730
        end = now - int(rng.integers(0, 30)) * SECONDS_PER_DAY
        start = end - span_days * SECONDS_PER_DAY
        offsets = np.sort(rng.integers(0, span_days * SECONDS_PER_DAY, size=k))
        timestamp = start + offsets

        parts.append(
            pd.DataFrame(
                {
                    "user_idx": user.user_idx,
                    "movie_idx": movie_idx,
                    "rating": rating,
                    "timestamp": timestamp,
                }
            )
        )

    ratings = pd.concat(parts, ignore_index=True)
    return ratings.sort_values(["user_idx", "timestamp"]).reset_index(drop=True)


def main() -> None:
    rng = np.random.default_rng(AZ_RANDOM_SEED)

    movies = load_and_filter_movies()
    movies = movies.reset_index(drop=True)
    movies["movie_idx"] = movies.index

    genre_matrix, genre_names = build_genre_matrix(movies)
    users = generate_users(rng, n_genres=len(genre_names))
    ratings = generate_ratings(rng, users, genre_matrix)

    movie_mapping = movies[["tmdb_id", "movie_idx"]]
    movies_out = movies[["movie_idx", "title", "genres", "poster_url"]].sort_values(
        "movie_idx"
    )

    user_mapping = pd.DataFrame(
        {"synthetic_user_id": users["user_idx"], "user_idx": users["user_idx"]}
    )
    users_out = users[
        ["user_idx", "gender", "age", "occupation", "zip_code"]
    ].sort_values("user_idx")

    ratings.to_parquet(AZ_PROCESSED_DIR / "ratings_clean.parquet", index=False)
    movies_out.to_parquet(AZ_PROCESSED_DIR / "movies.parquet", index=False)
    users_out.to_parquet(AZ_PROCESSED_DIR / "users.parquet", index=False)
    user_mapping.to_parquet(AZ_PROCESSED_DIR / "user_mapping.parquet", index=False)
    movie_mapping.to_parquet(AZ_PROCESSED_DIR / "movie_mapping.parquet", index=False)

    n_possible = len(users) * len(movies)
    print(
        f"\n{len(ratings):,} synthetic ratings, {len(users):,} synthetic users, "
        f"{len(movies):,} real films -> {AZ_PROCESSED_DIR}"
    )
    print(f"  sparsity: {(1 - len(ratings) / n_possible) * 100:.2f}% empty")
    print(f"  genres used for taste-matching: {len(genre_names)}")


if __name__ == "__main__":
    main()
