"""Fetch native Azerbaijani films from TMDB into raw/az/az_movies_raw.csv.
Requires TMDB_API_KEY (see datasets/.env.example)."""

import sys
import time

import pandas as pd
import requests
from tqdm import tqdm

from config import (
    AZ_MOVIES_RAW_PATH,
    AZ_TOP_N_MOVIES,
    TMDB_API_BASE,
    TMDB_API_KEY,
    TMDB_IMAGE_BASE,
)

REQUEST_DELAY_SECONDS = 0.05
DISCOVER_QUERIES = [
    {"with_original_language": "az"},
    {"with_origin_country": "AZ"},
]


def tmdb_get(session: requests.Session, path: str, params: dict) -> dict | None:
    params = {**params, "api_key": TMDB_API_KEY}
    for _attempt in range(3):
        resp = session.get(f"{TMDB_API_BASE}{path}", params=params, timeout=10)
        if resp.status_code == 429:
            time.sleep(float(resp.headers.get("Retry-After", 1)))
            continue
        resp.raise_for_status()
        return resp.json()
    return None


def fetch_genre_map(session: requests.Session) -> dict[int, str]:
    data = tmdb_get(session, "/genre/movie/list", {"language": "en-US"})
    return {g["id"]: g["name"] for g in (data or {}).get("genres", [])}


def discover_all(session: requests.Session, extra_params: dict) -> list[dict]:
    results = []
    page = 1
    total_pages = 1
    with tqdm(desc=f"discover {extra_params}") as bar:
        while page <= total_pages:
            data = tmdb_get(
                session, "/discover/movie", {**extra_params, "page": page}
            )
            if not data:
                break
            total_pages = data.get("total_pages", 1)
            results.extend(data.get("results", []))
            bar.total = total_pages
            bar.update(1)
            page += 1
            time.sleep(REQUEST_DELAY_SECONDS)
    return results


def main() -> None:
    if not TMDB_API_KEY:
        sys.exit(
            "TMDB_API_KEY is not set. Copy datasets/.env.example to datasets/.env "
            "and fill in your key, then re-run this script."
        )

    session = requests.Session()
    genre_map = fetch_genre_map(session)

    by_id: dict[int, dict] = {}
    for query in DISCOVER_QUERIES:
        for movie in discover_all(session, query):
            by_id.setdefault(movie["id"], movie)

    rows = []
    for movie in by_id.values():
        genre_names = [genre_map.get(gid) for gid in movie.get("genre_ids", [])]
        genre_names = [g for g in genre_names if g]
        release_date = movie.get("release_date") or ""
        year = int(release_date[:4]) if len(release_date) >= 4 else None
        poster_path = movie.get("poster_path")
        rows.append(
            {
                "tmdb_id": movie["id"],
                "title": movie.get("title") or movie.get("original_title"),
                "original_title": movie.get("original_title"),
                "year": year,
                "genres": "|".join(genre_names),
                "overview": movie.get("overview") or "",
                "poster_url": f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else None,
                "vote_average": movie.get("vote_average", 0.0),
                "vote_count": movie.get("vote_count", 0),
            }
        )

    out = pd.DataFrame(rows)
    n_fetched = len(out)

    out = out.assign(
        has_poster=out["poster_url"].notna(),
        has_overview=out["overview"].fillna("") != "",
    ).sort_values(
        ["vote_count", "vote_average", "has_poster", "has_overview"],
        ascending=False,
    )
    out = (
        out.head(AZ_TOP_N_MOVIES)
        .drop(columns=["has_poster", "has_overview"])
        .sort_values("tmdb_id")
        .reset_index(drop=True)
    )
    out.to_csv(AZ_MOVIES_RAW_PATH, index=False)

    print(
        f"\nFetched {n_fetched:,} unique Azerbaijani films, kept top {len(out):,} "
        f"by popularity -> {AZ_MOVIES_RAW_PATH}"
    )
    print(f"  with poster: {out['poster_url'].notna().sum():,}")
    print(f"  with overview: {(out['overview'] != '').sum():,}")
    print(f"  with genres: {(out['genres'] != '').sum():,}")
    print(f"  with any TMDB votes: {(out['vote_count'] > 0).sum():,}")


if __name__ == "__main__":
    main()
