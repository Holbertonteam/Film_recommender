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

# ml-1m ships without a movieId -> tmdbId mapping. We pull that mapping from
# ml-20m's links.csv instead: GroupLens keeps movieId stable across dataset
# versions, so ml-1m's ids are (almost entirely) a subset of ml-20m's, and
# ml-20m's catalog covers the pre-2000 films that make up ml-1m. We only ever
# need the one small file out of that archive.
ML_LINKS_SOURCE_URL = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
LINKS_CSV_PATH = RAW_DIR / "links.csv"

load_dotenv(DATASETS_DIR / ".env")
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_CACHE_PATH = RAW_DIR / "tmdb_cache.json"

MOVIES_ENRICHED_PATH = RAW_DIR / "movies_enriched.csv"

# Per-user temporal split: the most recent TEST_FRACTION of each user's
# ratings (by timestamp) become the test set, the rest are train.
TEST_FRACTION = 0.2
MIN_TEST_RATINGS_PER_USER = 1

# Env var opt-in, off by default: only set this if you've independently
# confirmed files.grouplens.org's TLS cert is the problem (not a
# man-in-the-middle) and you accept skipping verification for that one host.
ALLOW_INSECURE_MOVIELENS_DOWNLOAD = os.environ.get(
    "ALLOW_INSECURE_MOVIELENS_DOWNLOAD", ""
) == "1"
