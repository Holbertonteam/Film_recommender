"""Readers for the raw MovieLens 1M .dat files (shared by multiple scripts)."""

import pandas as pd

from config import ML_1M_DIR

RATINGS_COLS = ["user_id", "movie_id", "rating", "timestamp"]
USERS_COLS = ["user_id", "gender", "age", "occupation", "zip_code"]
MOVIES_COLS = ["movie_id", "title", "genres"]


def load_ratings() -> pd.DataFrame:
    return pd.read_csv(
        ML_1M_DIR / "ratings.dat",
        sep="::",
        engine="python",
        names=RATINGS_COLS,
        encoding="latin-1",
    )


def load_users() -> pd.DataFrame:
    return pd.read_csv(
        ML_1M_DIR / "users.dat",
        sep="::",
        engine="python",
        names=USERS_COLS,
        encoding="latin-1",
    )


def load_movies() -> pd.DataFrame:
    return pd.read_csv(
        ML_1M_DIR / "movies.dat",
        sep="::",
        engine="python",
        names=MOVIES_COLS,
        encoding="latin-1",
    )
