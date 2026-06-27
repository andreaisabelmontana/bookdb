"""The social-reading platform layer.

A thin SQLite-backed store for the things a reading platform tracks -- users,
books, ratings and shelves -- with a clean Python API on top that wires the
store into the collaborative-filtering engine:

    add_user / add_book / add_rating / shelve   -- write the social graph
    recommend_for(user)                          -- top-N CF recommendations
    kindred_readers(user)                        -- nearest users by taste
    similar_books(book)                          -- item-item neighbours

The recommender is rebuilt lazily from the current ratings whenever the data
changes, so reads always reflect the latest writes.
"""

from __future__ import annotations

import sqlite3
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from .matrix import RatingsMatrix
from .recommender import Recommender

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id   INTEGER PRIMARY KEY,
    name      TEXT NOT NULL,
    cluster   INTEGER
);
CREATE TABLE IF NOT EXISTS books (
    book_id   INTEGER PRIMARY KEY,
    title     TEXT NOT NULL,
    genre     TEXT,
    cluster   INTEGER
);
CREATE TABLE IF NOT EXISTS ratings (
    user_id   INTEGER NOT NULL,
    book_id   INTEGER NOT NULL,
    rating    REAL NOT NULL,
    PRIMARY KEY (user_id, book_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (book_id) REFERENCES books(book_id)
);
CREATE TABLE IF NOT EXISTS shelves (
    user_id   INTEGER NOT NULL,
    book_id   INTEGER NOT NULL,
    shelf     TEXT NOT NULL,          -- e.g. 'read', 'to-read', 'favourites'
    PRIMARY KEY (user_id, book_id, shelf),
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (book_id) REFERENCES books(book_id)
);
"""


class Platform:
    """SQLite-backed reading platform with a built-in recommender."""

    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        self._ratings_cache: Optional[RatingsMatrix] = None
        self._recommender: Optional[Recommender] = None

    # ---- context manager ------------------------------------------------

    def __enter__(self) -> "Platform":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    # ---- bulk load ------------------------------------------------------

    @classmethod
    def from_dataset(cls, dataset, db_path: str = ":memory:") -> "Platform":
        """Build a platform from a :class:`SyntheticDataset` (or equivalent)."""
        p = cls(db_path)
        p.conn.executemany(
            "INSERT OR REPLACE INTO users(user_id, name, cluster) VALUES (?,?,?)",
            dataset.users[["user_id", "name", "cluster"]].itertuples(
                index=False, name=None
            ),
        )
        p.conn.executemany(
            "INSERT OR REPLACE INTO books(book_id, title, genre, cluster) VALUES (?,?,?,?)",
            dataset.books[["book_id", "title", "genre", "cluster"]].itertuples(
                index=False, name=None
            ),
        )
        p.conn.executemany(
            "INSERT OR REPLACE INTO ratings(user_id, book_id, rating) VALUES (?,?,?)",
            dataset.ratings[["user_id", "book_id", "rating"]].itertuples(
                index=False, name=None
            ),
        )
        p.conn.commit()
        p._invalidate()
        return p

    # ---- writes ---------------------------------------------------------

    def add_user(self, user_id: int, name: str, cluster: Optional[int] = None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO users(user_id, name, cluster) VALUES (?,?,?)",
            (user_id, name, cluster),
        )
        self.conn.commit()

    def add_book(
        self, book_id: int, title: str, genre: Optional[str] = None,
        cluster: Optional[int] = None,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO books(book_id, title, genre, cluster) VALUES (?,?,?,?)",
            (book_id, title, genre, cluster),
        )
        self.conn.commit()

    def add_rating(self, user_id: int, book_id: int, rating: float) -> None:
        """Record (or overwrite) a user's rating of a book."""
        self.conn.execute(
            "INSERT OR REPLACE INTO ratings(user_id, book_id, rating) VALUES (?,?,?)",
            (user_id, book_id, float(rating)),
        )
        self.conn.commit()
        self._invalidate()

    def shelve(self, user_id: int, book_id: int, shelf: str = "read") -> None:
        """Add a book to one of the user's shelves."""
        self.conn.execute(
            "INSERT OR REPLACE INTO shelves(user_id, book_id, shelf) VALUES (?,?,?)",
            (user_id, book_id, shelf),
        )
        self.conn.commit()

    # ---- reads ----------------------------------------------------------

    def book_title(self, book_id: int) -> Optional[str]:
        row = self.conn.execute(
            "SELECT title FROM books WHERE book_id=?", (book_id,)
        ).fetchone()
        return row["title"] if row else None

    def user_name(self, user_id: int) -> Optional[str]:
        row = self.conn.execute(
            "SELECT name FROM users WHERE user_id=?", (user_id,)
        ).fetchone()
        return row["name"] if row else None

    def shelf(self, user_id: int, shelf: str = "read") -> List[int]:
        rows = self.conn.execute(
            "SELECT book_id FROM shelves WHERE user_id=? AND shelf=?",
            (user_id, shelf),
        ).fetchall()
        return [r["book_id"] for r in rows]

    def ratings_frame(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT user_id, book_id, rating FROM ratings", self.conn
        )

    # ---- recommender wiring --------------------------------------------

    def _invalidate(self) -> None:
        self._ratings_cache = None
        self._recommender = None

    def _matrix(self) -> RatingsMatrix:
        if self._ratings_cache is None:
            df = self.ratings_frame()
            if df.empty:
                raise ValueError("no ratings in the platform yet")
            self._ratings_cache = RatingsMatrix.from_dataframe(df)
        return self._ratings_cache

    def recommender(self) -> Recommender:
        if self._recommender is None:
            self._recommender = Recommender(self._matrix())
        return self._recommender

    def recommend_for(
        self, user_id: int, n: int = 10, method: str = "user", k: int = 20
    ) -> List[Tuple[int, float]]:
        """Top-``n`` recommended ``(book_id, score)`` for a user."""
        return self.recommender().recommend_for(user_id, n=n, method=method, k=k)

    def recommend_titles(
        self, user_id: int, n: int = 10, method: str = "user"
    ) -> List[Tuple[str, float]]:
        """Recommendations as ``(title, score)`` for human-readable output."""
        return [
            (self.book_title(bid) or f"book#{bid}", score)
            for bid, score in self.recommend_for(user_id, n=n, method=method)
        ]

    def kindred_readers(self, user_id: int, k: int = 5) -> List[Tuple[int, float]]:
        """The ``k`` readers most similar to ``user_id`` by taste."""
        return self.recommender().kindred_readers(user_id, k=k)

    def similar_books(self, book_id: int, n: int = 10) -> List[Tuple[int, float]]:
        return self.recommender().similar_books(book_id, n=n)
