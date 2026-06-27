"""Leave-one-out ranking evaluation.

For each user we hold out one of their *highly-rated* books, rebuild the
recommender without that rating, ask for a top-K list, and check whether the
held-out book comes back. Averaged over users this gives:

HitRate@K
    Fraction of held-out books that appear anywhere in the top-K list.

Recall@K
    Per-user recall of the held-out item; with exactly one held-out item per
    user this equals HitRate@K, but it is computed as a per-user mean so the
    definition generalizes to multiple held-out items.

Because the synthetic data has planted taste clusters, a personalized method
(user/item CF) should recover held-out books more often than the
non-personalized popularity baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .matrix import RatingsMatrix
from .recommender import Recommender


@dataclass
class EvalResult:
    method: str
    k: int
    hit_rate: float
    recall: float
    n_evaluated: int

    def __str__(self) -> str:
        return (
            f"{self.method:>10}  HitRate@{self.k}={self.hit_rate:.3f}  "
            f"Recall@{self.k}={self.recall:.3f}  (n={self.n_evaluated})"
        )


def leave_one_out(
    ratings: RatingsMatrix,
    method: str = "user",
    k: int = 10,
    *,
    min_rating: float = 4.0,
    max_users: Optional[int] = None,
    neighbours: int = 20,
    seed: int = 0,
) -> EvalResult:
    """Run leave-one-out evaluation for one recommender ``method``.

    Parameters
    ----------
    method : {"user", "item", "popularity"}
    k : int
        Length of the recommendation list.
    min_rating : float
        Only hold out books the user rated at least this highly (a "hit" should
        be a book they actually liked).
    max_users : int, optional
        Cap the number of users evaluated (for speed); chosen reproducibly.
    neighbours : int
        Neighbourhood size for user-based CF.
    """
    rng = np.random.default_rng(seed)
    user_ids = list(ratings.user_ids)
    if max_users is not None and max_users < len(user_ids):
        user_ids = list(rng.choice(user_ids, size=max_users, replace=False))

    hits = 0
    recalls: List[float] = []
    evaluated = 0

    for uid in user_ids:
        i = ratings.user_index[uid]
        row = ratings.matrix[i]
        cols = row.indices
        vals = row.data
        liked = cols[vals >= min_rating]
        if len(liked) < 2:
            continue  # need at least one to hold out and one to learn from
        held = ratings.book_ids[int(rng.choice(liked))]

        reduced = ratings.drop_rating(uid, held)
        rec = Recommender(reduced)
        top = rec.recommend_for(uid, n=k, method=method, k=neighbours)
        top_ids = {bid for bid, _ in top}

        evaluated += 1
        hit = held in top_ids
        hits += int(hit)
        recalls.append(1.0 if hit else 0.0)

    hit_rate = hits / evaluated if evaluated else 0.0
    recall = float(np.mean(recalls)) if recalls else 0.0
    return EvalResult(
        method=method, k=k, hit_rate=hit_rate, recall=recall, n_evaluated=evaluated
    )


def compare_methods(
    ratings: RatingsMatrix,
    k: int = 10,
    methods=("user", "item", "popularity"),
    **kwargs,
) -> Dict[str, EvalResult]:
    """Run leave-one-out for several methods and return a name -> result map."""
    return {m: leave_one_out(ratings, method=m, k=k, **kwargs) for m in methods}
