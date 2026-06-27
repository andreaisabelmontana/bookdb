"""User x book ratings matrix.

Builds a sparse CSR matrix from a long-form ``(user_id, book_id, rating)`` table
and keeps the bidirectional maps between external ids and internal matrix
indices. Everything downstream (similarity, recommendation, evaluation) operates
on the integer index space; the maps translate back to ids at the edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import sparse


def load_ratings(path: str) -> pd.DataFrame:
    """Read a ratings CSV with columns ``user_id,book_id,rating``."""
    df = pd.read_csv(path)
    expected = {"user_id", "book_id", "rating"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"ratings file missing columns: {sorted(missing)}")
    df = df[["user_id", "book_id", "rating"]].copy()
    df["rating"] = df["rating"].astype(float)
    return df


@dataclass
class RatingsMatrix:
    """A sparse user x book ratings matrix plus id<->index maps.

    Attributes
    ----------
    matrix : scipy.sparse.csr_matrix
        Shape ``(n_users, n_books)``; explicit zeros are treated as "unrated".
    user_ids / book_ids : list
        ``user_ids[i]`` is the external id of matrix row ``i`` (and likewise cols).
    """

    matrix: sparse.csr_matrix
    user_ids: List = field()
    book_ids: List = field()
    user_index: Dict = field(init=False)
    book_index: Dict = field(init=False)

    def __post_init__(self) -> None:
        self.user_index = {uid: i for i, uid in enumerate(self.user_ids)}
        self.book_index = {bid: j for j, bid in enumerate(self.book_ids)}

    # ---- construction ---------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        user_col: str = "user_id",
        book_col: str = "book_id",
        rating_col: str = "rating",
    ) -> "RatingsMatrix":
        """Build a matrix from a long-form ratings DataFrame.

        Duplicate ``(user, book)`` pairs are collapsed to their mean rating.
        """
        if df.empty:
            raise ValueError("cannot build a RatingsMatrix from an empty frame")

        agg = (
            df.groupby([user_col, book_col], as_index=False)[rating_col]
            .mean()
        )

        user_ids = sorted(agg[user_col].unique().tolist())
        book_ids = sorted(agg[book_col].unique().tolist())
        u_index = {uid: i for i, uid in enumerate(user_ids)}
        b_index = {bid: j for j, bid in enumerate(book_ids)}

        rows = agg[user_col].map(u_index).to_numpy()
        cols = agg[book_col].map(b_index).to_numpy()
        vals = agg[rating_col].to_numpy(dtype=float)

        mat = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(len(user_ids), len(book_ids))
        )
        return cls(matrix=mat, user_ids=user_ids, book_ids=book_ids)

    # ---- accessors ------------------------------------------------------

    @property
    def n_users(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_books(self) -> int:
        return self.matrix.shape[1]

    def rated_book_indices(self, user_id) -> np.ndarray:
        """Column indices of books this user has rated."""
        i = self.user_index[user_id]
        return self.matrix.indices[self.matrix.indptr[i] : self.matrix.indptr[i + 1]]

    def rated_book_ids(self, user_id) -> List:
        return [self.book_ids[j] for j in self.rated_book_indices(user_id)]

    def user_vector(self, user_id) -> np.ndarray:
        """Dense rating row for one user."""
        i = self.user_index[user_id]
        return np.asarray(self.matrix[i].todense()).ravel()

    def has_user(self, user_id) -> bool:
        return user_id in self.user_index

    def drop_rating(self, user_id, book_id) -> "RatingsMatrix":
        """Return a *new* matrix with one (user, book) entry removed.

        Used by leave-one-out evaluation. Ids and shape are preserved so index
        maps stay valid across the held-out fold.
        """
        i = self.user_index[user_id]
        j = self.book_index[book_id]
        lil = self.matrix.tolil()
        lil[i, j] = 0.0
        new = lil.tocsr()
        new.eliminate_zeros()
        return RatingsMatrix(
            matrix=new, user_ids=list(self.user_ids), book_ids=list(self.book_ids)
        )
