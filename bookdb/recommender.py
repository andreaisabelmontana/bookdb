"""Collaborative-filtering recommender.

Three strategies over the same ratings matrix:

user-based CF
    "Kindred readers" = the users most similar to the target by cosine
    similarity of their (mean-centered) rating vectors. A book's score for the
    target is the similarity-weighted average of how its kindred readers rated
    it. Formally, for target user ``u`` and book ``i``::

        score(u, i) = mu_u + ( sum_{v in N(u)} sim(u,v) * (r_{v,i} - mu_v) )
                             / sum_{v in N(u)} |sim(u,v)|

    where ``N(u)`` are the top-k most similar users who have rated ``i`` and
    ``mu`` are per-user mean ratings. Mean-centering removes the "harsh vs.
    generous rater" bias so similarity reflects taste, not scale.

item-based CF
    Symmetric: cosine similarity between *book* columns. A book scores high for
    a user when it is similar to the other books that user already rated highly::

        score(u, i) = ( sum_{j in rated(u)} sim(i,j) * r_{u,j} )
                       / sum_{j in rated(u)} |sim(i,j)|

    Co-loved books (rated together by the same readers) end up with high mutual
    similarity, so they get recommended together.

popularity baseline
    Non-personalized: rank books by a count-damped mean rating
    (``mean * n / (n + C)``), which pulls thinly-rated books toward the global
    mean. This is the bar collaborative filtering has to beat.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from .matrix import RatingsMatrix


class Recommender:
    def __init__(self, ratings: RatingsMatrix, *, shrink: float = 10.0):
        """Parameters
        ----------
        ratings : RatingsMatrix
        shrink : float
            Damping constant ``C`` for the popularity baseline.
        """
        self.ratings = ratings
        self.shrink = shrink
        self._user_means = self._compute_user_means()
        self._user_sim: Optional[np.ndarray] = None
        self._item_sim: Optional[np.ndarray] = None

    # ---- precomputation -------------------------------------------------

    def _compute_user_means(self) -> np.ndarray:
        m = self.ratings.matrix
        sums = np.asarray(m.sum(axis=1)).ravel()
        counts = np.diff(m.indptr)
        means = np.zeros_like(sums)
        nz = counts > 0
        means[nz] = sums[nz] / counts[nz]
        return means

    def _centered_matrix(self) -> sparse.csr_matrix:
        """Subtract each user's mean from their observed (nonzero) ratings."""
        m = self.ratings.matrix.tocoo()
        centered_vals = m.data - self._user_means[m.row]
        out = sparse.csr_matrix(
            (centered_vals, (m.row, m.col)), shape=m.shape
        )
        return out

    def user_similarity(self) -> np.ndarray:
        """Cached cosine similarity between users (on mean-centered rows)."""
        if self._user_sim is None:
            centered = self._centered_matrix()
            sim = cosine_similarity(centered, dense_output=True)
            np.fill_diagonal(sim, 0.0)
            self._user_sim = sim
        return self._user_sim

    def item_similarity(self) -> np.ndarray:
        """Cached cosine similarity between books.

        Computed on *mean-centered* columns (adjusted cosine): each book column
        has the global book mean subtracted from its observed entries before the
        cosine. Centering lets oppositely-loved books take a negative similarity
        instead of an artificially positive one, so a book the user dislikes does
        not pull its anti-correlated neighbours up.
        """
        if self._item_sim is None:
            m = self.ratings.matrix.tocoo()
            sums = np.asarray(self.ratings.matrix.sum(axis=0)).ravel()
            counts = np.asarray((self.ratings.matrix != 0).sum(axis=0)).ravel()
            book_means = np.divide(
                sums, counts, out=np.zeros_like(sums), where=counts > 0
            )
            centered_vals = m.data - book_means[m.col]
            centered = sparse.csr_matrix(
                (centered_vals, (m.row, m.col)), shape=m.shape
            )
            sim = cosine_similarity(centered.T, dense_output=True)
            np.fill_diagonal(sim, 0.0)
            self._item_sim = sim
        return self._item_sim

    # ---- kindred readers ------------------------------------------------

    def kindred_readers(self, user_id, k: int = 5) -> List[Tuple[object, float]]:
        """Top-``k`` most similar users to ``user_id`` as ``(id, similarity)``.

        Only positive-similarity neighbours are returned (negative cosine means
        opposite taste, which we don't treat as "kindred").
        """
        if not self.ratings.has_user(user_id):
            raise KeyError(f"unknown user: {user_id}")
        i = self.ratings.user_index[user_id]
        sim_row = self.user_similarity()[i]
        order = np.argsort(sim_row)[::-1]
        out: List[Tuple[object, float]] = []
        for j in order:
            if sim_row[j] <= 0:
                break
            out.append((self.ratings.user_ids[j], float(sim_row[j])))
            if len(out) >= k:
                break
        return out

    # ---- scoring --------------------------------------------------------

    def _user_based_scores(self, user_id, k: int, min_support: int = 2) -> np.ndarray:
        i = self.ratings.user_index[user_id]
        sim_row = self.user_similarity()[i].copy()
        sim_row[sim_row < 0] = 0.0
        # keep only the top-k positive neighbours
        if k < len(sim_row):
            cutoff_idx = np.argpartition(sim_row, -k)[-k:]
            mask = np.zeros_like(sim_row, dtype=bool)
            mask[cutoff_idx] = True
            sim_row[~mask] = 0.0

        centered = self._centered_matrix()  # users x books, mean-centered
        # weighted sum over neighbours of their centered ratings
        weighted = sim_row @ centered                      # (n_books,)
        weighted = np.asarray(weighted).ravel()
        # denominator: sum of |sim| over neighbours who actually rated each book
        rated_mask = (self.ratings.matrix != 0).astype(float)
        denom = sim_row @ rated_mask
        denom = np.asarray(denom).ravel()
        # how many of the kept neighbours actually rated each book (support)
        neighbour_mask = (sim_row > 0).astype(float)
        support = neighbour_mask @ rated_mask
        support = np.asarray(support).ravel()

        scores = np.full(self.ratings.n_books, -np.inf, dtype=float)
        # only score books with enough neighbour evidence; a single noisy
        # neighbour shouldn't be able to float a book to the top of the list.
        ok = (denom > 1e-12) & (support >= min_support)
        scores[ok] = self._user_means[i] + weighted[ok] / denom[ok]
        return scores

    def _item_based_scores(self, user_id) -> np.ndarray:
        u = self.ratings.user_index[user_id]
        sim = self.item_similarity()                        # books x books
        user_row = self.ratings.user_vector(user_id)        # (n_books,)
        rated = user_row != 0
        if not rated.any():
            return np.full(self.ratings.n_books, -np.inf)
        weighted = sim[:, rated] @ user_row[rated]          # (n_books,)
        denom = np.abs(sim[:, rated]).sum(axis=1)
        scores = np.full(self.ratings.n_books, -np.inf, dtype=float)
        nz = denom > 1e-12
        scores[nz] = weighted[nz] / denom[nz]
        return scores

    def popularity_scores(self) -> np.ndarray:
        """Count-damped mean rating per book (the non-personalized baseline)."""
        m = self.ratings.matrix
        sums = np.asarray(m.sum(axis=0)).ravel()
        counts = np.asarray((m != 0).sum(axis=0)).ravel()
        global_mean = sums.sum() / max(counts.sum(), 1)
        means = np.where(counts > 0, sums / np.maximum(counts, 1), global_mean)
        # shrink thinly-rated books toward the global mean
        return (counts * means + self.shrink * global_mean) / (counts + self.shrink)

    # ---- recommendation -------------------------------------------------

    def recommend_for(
        self,
        user_id,
        n: int = 10,
        method: str = "user",
        k: int = 20,
    ) -> List[Tuple[object, float]]:
        """Top-``n`` book recommendations for a user, excluding already-read.

        Parameters
        ----------
        method : {"user", "item", "popularity"}
        k : int
            Neighbourhood size for user-based CF.
        """
        if method == "popularity":
            scores = self.popularity_scores()
        elif not self.ratings.has_user(user_id):
            # cold start -> fall back to popularity
            scores = self.popularity_scores()
        elif method == "user":
            scores = self._user_based_scores(user_id, k)
        elif method == "item":
            scores = self._item_based_scores(user_id)
        else:
            raise ValueError(f"unknown method: {method!r}")

        scores = scores.copy()
        if self.ratings.has_user(user_id):
            already = self.ratings.rated_book_indices(user_id)
            scores[already] = -np.inf

        n = min(n, int(np.isfinite(scores).sum()))
        if n <= 0:
            return []
        top = np.argpartition(scores, -n)[-n:]
        top = top[np.argsort(scores[top])[::-1]]
        return [(self.ratings.book_ids[j], float(scores[j])) for j in top]

    def similar_books(self, book_id, n: int = 10) -> List[Tuple[object, float]]:
        """Top-``n`` books most similar to ``book_id`` (item-item cosine)."""
        if book_id not in self.ratings.book_index:
            raise KeyError(f"unknown book: {book_id}")
        j = self.ratings.book_index[book_id]
        sim_row = self.item_similarity()[j]
        order = np.argsort(sim_row)[::-1][:n]
        return [(self.ratings.book_ids[c], float(sim_row[c])) for c in order]
