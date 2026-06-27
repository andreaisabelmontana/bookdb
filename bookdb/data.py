"""Synthetic ratings dataset with planted latent taste clusters.

Each reader belongs to one of ``n_clusters`` taste groups; each book belongs to
one genre that aligns with a cluster. A reader rates a book highly when the
book's genre matches their taste and low otherwise, with Gaussian noise. This
plants exactly the structure collaborative filtering is meant to discover:
readers in the same cluster are "kindred", and books in the same genre are
"co-loved". The popularity baseline, which ignores who is rating, should be
beatable on this data.

Deterministic given ``seed`` so the committed CSV and the tests agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class SyntheticDataset:
    ratings: pd.DataFrame          # user_id, book_id, rating
    books: pd.DataFrame            # book_id, title, genre, cluster
    users: pd.DataFrame           # user_id, name, cluster
    n_clusters: int


# A few evocative title fragments per genre so the demo reads like real books.
_GENRE_NAMES = [
    "Literary Fiction",
    "Hard Science Fiction",
    "Cozy Mystery",
    "Epic Fantasy",
    "Narrative History",
    "Speculative Horror",
]
_TITLE_LEFT = [
    "The", "A", "Last", "Quiet", "Burning", "Hidden", "Distant", "Crimson",
    "Hollow", "Silver", "Broken", "Northern", "Pale", "Wandering", "Final",
]
_TITLE_RIGHT = [
    "Garden", "Engine", "Lighthouse", "Cartographer", "Orchard", "Cipher",
    "Migration", "Observatory", "Tide", "Inheritance", "Archive", "Reckoning",
    "Meridian", "Harvest", "Threshold", "Confession", "Atlas", "Hourglass",
]
_FIRST = [
    "Ada", "Ben", "Cleo", "Dario", "Esme", "Finn", "Greta", "Hugo", "Iris",
    "Jonas", "Kira", "Liam", "Mara", "Noor", "Otto", "Petra", "Quinn", "Rafa",
    "Sena", "Theo", "Uma", "Vera", "Wes", "Xan", "Yara", "Zane",
]


def generate_synthetic(
    n_users: int = 200,
    n_books: int = 120,
    n_clusters: int = 4,
    ratings_per_user: int = 18,
    noise: float = 0.6,
    seed: int = 7,
) -> SyntheticDataset:
    """Generate a clustered synthetic ratings dataset.

    Parameters
    ----------
    n_users, n_books : int
    n_clusters : int
        Number of latent taste groups (and book genres used, capped at this).
    ratings_per_user : int
        Approximate number of books each user rates (sparsity control).
    noise : float
        Std-dev of Gaussian rating noise.
    seed : int
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    n_genres = min(n_clusters, len(_GENRE_NAMES))

    user_cluster = rng.integers(0, n_clusters, size=n_users)
    book_cluster = rng.integers(0, n_genres, size=n_books)

    # --- book / user metadata -------------------------------------------
    titles = _unique_titles(rng, n_books)
    books = pd.DataFrame(
        {
            "book_id": np.arange(n_books),
            "title": titles,
            "genre": [_GENRE_NAMES[c] for c in book_cluster],
            "cluster": book_cluster,
        }
    )
    users = pd.DataFrame(
        {
            "user_id": np.arange(n_users),
            "name": [f"{_FIRST[i % len(_FIRST)]}-{i:03d}" for i in range(n_users)],
            "cluster": user_cluster,
        }
    )

    # --- ratings ---------------------------------------------------------
    # Bias each user toward sampling books from their own cluster, so cluster
    # mates share overlap (kindred) and produce co-rated books (co-loved).
    records = []
    book_idx_by_cluster = {
        c: np.where(book_cluster == (c % n_genres))[0] for c in range(n_clusters)
    }
    all_books = np.arange(n_books)
    for u in range(n_users):
        c = user_cluster[u]
        in_cluster = book_idx_by_cluster[c]
        n_in = min(len(in_cluster), int(ratings_per_user * 0.7))
        chosen_in = rng.choice(in_cluster, size=n_in, replace=False)
        n_out = ratings_per_user - n_in
        chosen_out = rng.choice(all_books, size=n_out, replace=False)
        chosen = np.unique(np.concatenate([chosen_in, chosen_out]))
        for b in chosen:
            match = (user_cluster[u] % n_genres) == book_cluster[b]
            base = 4.4 if match else 2.2
            r = base + rng.normal(0.0, noise)
            r = float(np.clip(round(r * 2) / 2, 1.0, 5.0))  # 1..5 in 0.5 steps
            records.append((u, int(b), r))

    ratings = pd.DataFrame(records, columns=["user_id", "book_id", "rating"])
    ratings = ratings.drop_duplicates(subset=["user_id", "book_id"]).reset_index(
        drop=True
    )
    return SyntheticDataset(
        ratings=ratings, books=books, users=users, n_clusters=n_clusters
    )


def _unique_titles(rng: np.random.Generator, n: int) -> List[str]:
    seen = set()
    out: List[str] = []
    while len(out) < n:
        t = f"{rng.choice(_TITLE_LEFT)} {rng.choice(_TITLE_RIGHT)}"
        key = t
        bump = 0
        while key in seen:
            bump += 1
            key = f"{t} {_roman(bump)}"
        seen.add(key)
        out.append(key)
    return out


def _roman(n: int) -> str:
    table = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for v, s in table:
        while n >= v:
            out += s
            n -= v
    return out
