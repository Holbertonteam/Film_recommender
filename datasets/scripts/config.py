"""Shared paths, constants, and env loading for the CineMatch data pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv

DATASETS_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = DATASETS_DIR / "raw"
PROCESSED_DIR = DATASETS_DIR / "processed"
FIGURES_DIR = DATASETS_DIR / "figures"

for _dir in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

ML_1M_DIR = RAW_DIR / "ml-1m"
ML_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

# ml-1m has no movieId -> tmdbId mapping, so we pull it from ml-20m's links.csv
ML_LINKS_SOURCE_URL = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
LINKS_CSV_PATH = RAW_DIR / "links.csv"

load_dotenv(DATASETS_DIR / ".env")
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_CACHE_PATH = RAW_DIR / "tmdb_cache.json"

MOVIES_ENRICHED_PATH = RAW_DIR / "movies_enriched.csv"

# most recent TEST_FRACTION of each user's ratings become the test set
TEST_FRACTION = 0.2
MIN_TEST_RATINGS_PER_USER = 1

# Azerbaijani-films pipeline (az_*.py scripts)
AZ_RAW_DIR = RAW_DIR / "az"
AZ_PROCESSED_DIR = DATASETS_DIR / "processed_az"
AZ_MOVIES_RAW_PATH = AZ_RAW_DIR / "az_movies_raw.csv"

for _dir in (AZ_RAW_DIR, AZ_PROCESSED_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

AZ_RANDOM_SEED = 42
AZ_TOP_N_MOVIES = 100
AZ_N_SYNTHETIC_USERS = 400
AZ_MIN_RATINGS_PER_USER = 15
AZ_MAX_RATINGS_PER_USER = 60

# opt-in only, see net.py
ALLOW_INSECURE_MOVIELENS_DOWNLOAD = os.environ.get(
    "ALLOW_INSECURE_MOVIELENS_DOWNLOAD", ""
) == "1"
