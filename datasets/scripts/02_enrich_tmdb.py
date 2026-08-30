"""Enrich MovieLens 1M movies with TMDB posters and genres.

Matching strategy:
  1. Primary: join on the movieId -> tmdbId mapping from ml-20m's links.csv
     (GroupLens keeps movieId stable across dataset versions, and ml-20m's
     catalog covers the pre-2000 films that make up ml-1m).
  2. Fallback: for any movie without a links.csv mapping (or where the TMDB
     id it names 404s), search TMDB by parsed title + release year and take
     the top hit.

Requires TMDB_API_KEY (see datasets/.env.example). Results are cached in
raw/tmdb_cache.json so re-runs don't re-hit the API, and a partial run can be
safely re-run to pick up where it left off.

Output: raw/movies_enriched.csv (movie_id, title, year, genres, poster_url,
matched_via) -- an intermediate file consumed by 03_clean_and_index.py.
"""

import json
import re
import sys
import time
import zipfile

import pandas as pd
import requests
from tqdm import tqdm

from config import (
    LINKS_CSV_PATH,
    ML_LINKS_SOURCE_URL,
    MOVIES_ENRICHED_PATH,
    RAW_DIR,
    TMDB_API_BASE,
    TMDB_API_KEY,
    TMDB_CACHE_PATH,
    TMDB_IMAGE_BASE,
)
from movielens_io import load_movies
from net import download_file

TITLE_YEAR_RE = re.compile(r"^(.*)\s\((\d{4})\)$")
REQUEST_DELAY_SECONDS = 0.05
CACHE_SAVE_EVERY = 50


def ensure_links_csv() -> None:
    if LINKS_CSV_PATH.exists():
        print(f"Found existing {LINKS_CSV_PATH}, skipping links source download.")
        return

    print(
        "Fetching ml-20m's links.csv (movieId -> tmdbId mapping) -- this means "
        "downloading the full ml-20m archive once, but only links.csv is kept."
    )
    tmp_zip = RAW_DIR / "_ml-links-source-tmp.zip"
    download_file(ML_LINKS_SOURCE_URL, tmp_zip, description="ml-20m.zip (for links.csv)")

    with zipfile.ZipFile(tmp_zip) as zf:
        member = next(n for n in zf.namelist() if n.endswith("links.csv"))
        with zf.open(member) as src, open(LINKS_CSV_PATH, "wb") as dst:
            dst.write(src.read())
    tmp_zip.unlink()


def load_links() -> pd.DataFrame:
    links = pd.read_csv(LINKS_CSV_PATH)
    links = links.rename(columns={"movieId": "movie_id", "tmdbId": "tmdb_id"})
    return links[["movie_id", "tmdb_id"]]


def parse_title_year(raw_title: str) -> tuple[str, int | None]:
    m = TITLE_YEAR_RE.match(raw_title.strip())
    if not m:
        return raw_title.strip(), None
    return m.group(1).strip(), int(m.group(2))


def load_cache() -> dict:
    if TMDB_CACHE_PATH.exists():
        with open(TMDB_CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    with open(TMDB_CACHE_PATH, "w") as f:
        json.dump(cache, f)


def tmdb_get(session: requests.Session, path: str, params: dict) -> dict | None:
    params = {**params, "api_key": TMDB_API_KEY}
    for attempt in range(3):
        resp = session.get(f"{TMDB_API_BASE}{path}", params=params, timeout=10)
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", 1))
            time.sleep(wait)
            continue
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    return None


def fetch_by_tmdb_id(session: requests.Session, tmdb_id: int) -> dict | None:
    return tmdb_get(session, f"/movie/{tmdb_id}", {})


def search_by_title_year(
    session: requests.Session, title: str, year: int | None
) -> dict | None:
    params = {"query": title}
    if year is not None:
        params["year"] = year
    result = tmdb_get(session, "/search/movie", params)
    if not result or not result.get("results"):
        return None
    top = result["results"][0]
    return fetch_by_tmdb_id(session, top["id"])


def main() -> None:
    if not TMDB_API_KEY:
        sys.exit(
            "TMDB_API_KEY is not set. Copy datasets/.env.example to datasets/.env "
            "and fill in your key (see datasets/README.md for how to get one), "
            "then re-run this script."
        )

    ensure_links_csv()
    movies = load_movies()
    links = load_links()
    movies = movies.merge(links, on="movie_id", how="left")

    parsed = movies["title"].map(parse_title_year)
    movies["clean_title"] = [p[0] for p in parsed]
    movies["year"] = [p[1] for p in parsed]

    cache = load_cache()
    session = requests.Session()
    matched_via_counts = {"links": 0, "search": 0, "none": 0}

    for i, row in enumerate(tqdm(movies.itertuples(), total=len(movies), desc="TMDB enrich")):
        key = str(row.movie_id)
        if key in cache:
            continue

        details = None
        matched_via = "none"

        if pd.notna(row.tmdb_id):
            details = fetch_by_tmdb_id(session, int(row.tmdb_id))
            if details is not None:
                matched_via = "links"
            time.sleep(REQUEST_DELAY_SECONDS)

        if details is None:
            details = search_by_title_year(session, row.clean_title, row.year)
            if details is not None:
                matched_via = "search"
            time.sleep(REQUEST_DELAY_SECONDS)

        if details is not None:
            cache[key] = {
                "poster_path": details.get("poster_path"),
                "genres": [g["name"] for g in details.get("genres", [])],
                "matched_via": matched_via,
            }
        else:
            cache[key] = {"poster_path": None, "genres": [], "matched_via": "none"}

        if (i + 1) % CACHE_SAVE_EVERY == 0:
            save_cache(cache)

    save_cache(cache)

    def tmdb_genres(movie_id: int) -> str | None:
        genres = cache.get(str(movie_id), {}).get("genres") or []
        return "|".join(genres) if genres else None

    def poster_url(movie_id: int) -> str | None:
        path = cache.get(str(movie_id), {}).get("poster_path")
        return f"{TMDB_IMAGE_BASE}{path}" if path else None

    movies["genres_tmdb"] = movies["movie_id"].map(tmdb_genres)
    movies["genres"] = movies["genres_tmdb"].fillna(movies["genres"])
    movies["poster_url"] = movies["movie_id"].map(poster_url)
    movies["matched_via"] = movies["movie_id"].map(
        lambda mid: cache.get(str(mid), {}).get("matched_via", "none")
    )

    for mid in movies["movie_id"]:
        matched_via_counts[cache.get(str(mid), {}).get("matched_via", "none")] += 1

    out = movies[
        ["movie_id", "clean_title", "year", "genres", "poster_url", "matched_via"]
    ].rename(columns={"clean_title": "title"})
    out.to_csv(MOVIES_ENRICHED_PATH, index=False)

    n = len(movies)
    print(f"\nEnriched {n:,} movies -> {MOVIES_ENRICHED_PATH}")
    print(f"  matched via links.csv: {matched_via_counts['links']:,}")
    print(f"  matched via title/year search: {matched_via_counts['search']:,}")
    print(f"  unmatched (no poster/genres from TMDB): {matched_via_counts['none']:,}")
    print(f"  poster coverage: {out['poster_url'].notna().mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
