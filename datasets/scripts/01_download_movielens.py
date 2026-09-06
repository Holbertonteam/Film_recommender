"""Download and unpack the MovieLens 1M dataset into datasets/raw/ml-1m/.

Safe to re-run: skips the download if the files are already present.
"""

import zipfile

from config import ML_1M_DIR, ML_1M_URL, RAW_DIR
from movielens_io import load_movies, load_ratings, load_users
from net import download_file


def ensure_movielens_1m() -> None:
    if (ML_1M_DIR / "ratings.dat").exists():
        print(f"Found existing dataset at {ML_1M_DIR}, skipping download.")
        return

    zip_path = RAW_DIR / "ml-1m.zip"
    download_file(ML_1M_URL, zip_path, description="ml-1m.zip")

    print(f"Extracting to {RAW_DIR} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    zip_path.unlink()


def main() -> None:
    ensure_movielens_1m()

    ratings = load_ratings()
    users = load_users()
    movies = load_movies()

    print("\nMovieLens 1M loaded:")
    print(f"  ratings: {len(ratings):,} rows")
    print(f"  users:   {len(users):,} rows")
    print(f"  movies:  {len(movies):,} rows")


if __name__ == "__main__":
    main()
