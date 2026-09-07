"""Shared HTTP download helper."""

import sys

import requests
from tqdm import tqdm

from config import ALLOW_INSECURE_MOVIELENS_DOWNLOAD


def download_file(url: str, dest_path, description: str | None = None) -> None:
    label = description or dest_path.name
    verify = True
    try:
        _stream_download(url, dest_path, label, verify=verify)
    except requests.exceptions.SSLError:
        if not ALLOW_INSECURE_MOVIELENS_DOWNLOAD:
            print(
                f"\nSSL verification failed while downloading {url}\n"
                "This has been observed with files.grouplens.org's certificate "
                "and is NOT something this script should silently work around, "
                "since disabling verification also hides a real "
                "man-in-the-middle attack.\n\n"
                "Options:\n"
                "  1) Check https://grouplens.org/datasets/movielens/ for a status "
                "notice, and retry later once the cert is renewed.\n"
                "  2) Download the file yourself in a browser and place it at:\n"
                f"       {dest_path}\n"
                "     then re-run this script (it will detect the file and skip "
                "downloading).\n"
                "  3) If you've independently confirmed this is the known expired-cert "
                "issue and accept the risk, opt in explicitly:\n"
                "       export ALLOW_INSECURE_MOVIELENS_DOWNLOAD=1\n",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            f"WARNING: retrying {url} with TLS verification disabled "
            "(ALLOW_INSECURE_MOVIELENS_DOWNLOAD=1).",
            file=sys.stderr,
        )
        _stream_download(url, dest_path, label, verify=False)


def _stream_download(url: str, dest_path, label: str, verify: bool) -> None:
    with requests.get(url, stream=True, timeout=30, verify=verify) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
        with open(tmp_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=label
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                bar.update(len(chunk))
        tmp_path.rename(dest_path)
