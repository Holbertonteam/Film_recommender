# CineMatch data pipeline

Everything upstream of the model: download raw MovieLens data, enrich it with
TMDB posters/genres, clean and re-index it, split it temporally, and produce
the EDA charts + the files the PyTorch training code reads.

## Setup

```bash
cd datasets
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

cp .env.example .env
# edit .env, set TMDB_API_KEY (get a free key at https://www.themoviedb.org/settings/api
# -> API -> Create -> API Read Access Token page also shows the v3 "API Key")
```

`.env` is git-ignored. Never commit your key.

## Running it

```bash
cd datasets/scripts
python3 run_pipeline.py
```

This runs all five steps in order. Re-running is safe and mostly incremental:
already-downloaded raw files and already-cached TMDB lookups are skipped/reused.

You can also run steps individually (useful while iterating, or to skip TMDB
enrichment while you're just testing the cleaning/split/EDA logic):

| # | Script | What it does |
|---|--------|---------------|
| 1 | `01_download_movielens.py` | Downloads + unpacks MovieLens 1M into `raw/ml-1m/` |
| 2 | `02_enrich_tmdb.py` | Adds TMDB posters + genres, writes `raw/movies_enriched.csv` |
| 3 | `03_clean_and_index.py` | Cleans data, maps raw ids to contiguous indices, writes the `processed/` tables |
| 4 | `04_temporal_split.py` | Per-user temporal train/test split, writes `ratings_train.parquet` / `ratings_test.parquet` |
| 5 | `05_eda.py` | Sparsity / rating distribution / long-tail charts, writes `figures/*.png` |

Each numbered script can be run directly, e.g. `python3 03_clean_and_index.py`,
as long as the previous step's output already exists.

## Output files (what your training code should read)

All in `datasets/processed/`, as Parquet (`pd.read_parquet`):

**`ratings_train.parquet` / `ratings_test.parquet`** — the files to train and
evaluate on. Schema:

| column | type | meaning |
|---|---|---|
| `user_idx` | int64 | 0-based contiguous user index, for `nn.Embedding` |
| `movie_idx` | int64 | 0-based contiguous movie index, for `nn.Embedding` |
| `rating` | int64 | 1-5 stars |
| `timestamp` | int64 | Unix seconds |

**`movies.parquet`** — one row per movie that appears in the ratings:

| column | type | meaning |
|---|---|---|
| `movie_idx` | int64 | matches `movie_idx` above |
| `title` | string | title, year stripped out (see below) |
| `genres` | string | `\|`-separated genre names (TMDB genres where a match was found, MovieLens' own genre list otherwise) |
| `poster_url` | string or null | full TMDB image URL; **null when no TMDB match was found** — render a placeholder for these |

Also written, as reference/reproducibility tables (not needed by the core
training loop, but here if you need to trace an idx back to the raw MovieLens
id, or want the demographic fields):

- `user_mapping.parquet`: `user_id` (raw MovieLens id) -> `user_idx`
- `movie_mapping.parquet`: `movie_id` (raw MovieLens id) -> `movie_idx`
- `users.parquet`: `user_idx, gender, age, occupation, zip_code` (MovieLens' own demographic fields, bonus data)
- `ratings_clean.parquet`: the full cleaned+indexed ratings table *before* the train/test split (all of train + test combined) — use this if you want to do your own split instead of the temporal one

`datasets/figures/*.png` are the EDA charts (sparsity, rating distribution,
long-tail popularity), sized and styled to drop straight into slides.

## Design notes

**Temporal split, not random.** For each user, ratings are sorted by
timestamp and the most recent 20% become the test set (`TEST_FRACTION` in
`scripts/config.py`), the rest are train. This mirrors how the model will
actually be used — predicting a user's *future* ratings from their *past*
ones — so it doesn't leak future information into training the way a random
split would. Every user in ml-1m has >=20 ratings, so this always leaves at
least a few ratings on each side.

**TMDB matching.** MovieLens 1M's `movies.dat` has no TMDB id, so matching
happens in two passes: (1) join on `movieId` against ml-20m's `links.csv`
(GroupLens keeps movie ids stable across dataset versions, and ml-20m's
catalog covers the pre-2000 films ml-1m consists of), which is exact when it
hits; (2) for anything that misses, fall back to a TMDB title+year search and
take the top result. `02_enrich_tmdb.py` prints a match-rate summary
(links / search / unmatched) each run. TMDB responses are cached in
`raw/tmdb_cache.json`, so re-running only fetches movies that failed last
time.

**Known limitations:**
- A small number of movies won't match on TMDB at all (obscure titles, or a
  title+year search that doesn't find the right film) — these get
  `poster_url = null`, falling back to MovieLens' own genre list.
- The temporal split doesn't specially handle ties: if a user rated multiple
  movies in the exact same second at the train/test boundary, they could
  split across it. `04_temporal_split.py` flags how many users this affects;
  it's rare with ml-1m's per-second timestamps.

## Troubleshooting

**`SSLError` / "certificate has expired" downloading from
`files.grouplens.org`.** This is a known issue with GroupLens' own server
certificate, not your machine or this script. The script deliberately does
**not** silently disable TLS verification (that would also hide a genuine
man-in-the-middle attack). Options, in order of preference:
1. Wait and retry — it's usually a matter of their ops team renewing the cert.
2. Download the zip yourself in a browser from
   https://grouplens.org/datasets/movielens/1m/ (or the equivalent ml-20m
   link for `links.csv`) and place it where the error message says; the
   script detects the file and skips the download.
3. If you've looked at the certificate yourself and are confident it's this
   known expired-cert issue (not a MITM), you can opt in to skipping
   verification for just that request: `export
   ALLOW_INSECURE_MOVIELENS_DOWNLOAD=1`.

**`TMDB_API_KEY is not set`** — copy `.env.example` to `.env` in this
directory and fill in your key.

**Rate limits / slow enrichment** — `02_enrich_tmdb.py` is resumable; if it's
interrupted, just re-run it and it'll pick up from the cache.
