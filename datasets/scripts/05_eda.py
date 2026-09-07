"""Exploratory analysis: sparsity, rating distribution, long-tail popularity.
Saves one PNG per chart to datasets/figures/."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURES_DIR, PROCESSED_DIR
from style import BLUE, BLUE_RAMP, INK_MUTED, INK_PRIMARY, apply_style


MIN_LABEL_WIDTH_PCT = 12  # below this segment width, the label can't fit inside


def _label_segment(ax, left: float, width: float, text: str, inside_color: str) -> None:
    center = left + width / 2
    if width >= MIN_LABEL_WIDTH_PCT:
        ax.text(
            center, 0, text, ha="center", va="center",
            color=inside_color, fontsize=10, fontweight="bold",
        )
    else:
        ax.plot([center, center], [0.25, 0.32], color=INK_MUTED, linewidth=1)
        ax.text(
            center, 0.35, text, ha="center", va="bottom",
            color=INK_PRIMARY, fontsize=9,
        )


def plot_sparsity(ratings: pd.DataFrame) -> None:
    n_users = ratings["user_idx"].nunique()
    n_movies = ratings["movie_idx"].nunique()
    n_ratings = len(ratings)
    possible = n_users * n_movies
    density_pct = 100 * n_ratings / possible
    sparsity_pct = 100 - density_pct

    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.barh([0], [density_pct], color=BLUE, height=0.5, label="Rated")
    ax.barh([0], [sparsity_pct], left=[density_pct], color="#e1e0d9", height=0.5, label="Empty")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.7)
    ax.set_yticks([])
    ax.set_xlabel("% of user-movie matrix")
    ax.grid(False)
    ax.set_title("User-Movie Matrix Sparsity")
    _label_segment(ax, 0, density_pct, f"{density_pct:.2f}% rated", "white")
    _label_segment(ax, density_pct, sparsity_pct, f"{sparsity_pct:.1f}% empty", INK_PRIMARY)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_sparsity.png")
    plt.close(fig)

    print(f"Sparsity: {n_users:,} users x {n_movies:,} movies = {possible:,} possible "
          f"ratings; {n_ratings:,} actual -> {sparsity_pct:.2f}% empty "
          f"({density_pct:.2f}% dense).")


def plot_rating_distribution(ratings: pd.DataFrame) -> None:
    counts = ratings["rating"].value_counts().sort_index()
    pct = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=BLUE_RAMP, width=0.6)
    for bar, p in zip(bars, pct):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{p:.1f}%", ha="center", va="bottom", color=INK_PRIMARY, fontsize=9,
        )
    ax.set_xticks(counts.index)
    ax.set_xlabel("Rating (stars)")
    ax.set_ylabel("Number of ratings")
    ax.set_title("Rating Distribution")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_rating_distribution.png")
    plt.close(fig)

    print("Rating distribution: " + ", ".join(f"{r}*={p:.1f}%" for r, p in pct.items()))


def plot_long_tail(ratings: pd.DataFrame) -> None:
    counts = ratings["movie_idx"].value_counts().sort_values(ascending=False)
    ranks = np.arange(1, len(counts) + 1)

    cutoff = max(1, round(0.2 * len(counts)))
    top20_share = counts.iloc[:cutoff].sum() / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ranks, counts.values, color=BLUE, linewidth=2)
    ax.set_yscale("log")
    ax.axvline(cutoff, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.text(
        cutoff, counts.iloc[0], f"  top 20% of movies\n  = {top20_share:.0f}% of ratings",
        va="top", color=INK_PRIMARY, fontsize=9,
    )
    ax.set_xlabel("Movie popularity rank")
    ax.set_ylabel("Number of ratings (log scale)")
    ax.set_title("Long-Tail Movie Popularity")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_long_tail_popularity.png")
    plt.close(fig)

    print(f"Long tail: top 20% of movies account for {top20_share:.1f}% of all ratings.")


def main() -> None:
    apply_style()
    ratings = pd.read_parquet(PROCESSED_DIR / "ratings_clean.parquet")

    plot_sparsity(ratings)
    plot_rating_distribution(ratings)
    plot_long_tail(ratings)

    print(f"\nCharts saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
