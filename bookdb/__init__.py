"""BookDB: a book recommendation engine wrapped in a social reading platform.

Public API:
    Recommender          -- user-based & item-based collaborative filtering + popularity baseline
    RatingsMatrix        -- sparse user x book ratings matrix with id <-> index maps
    Platform             -- SQLite-backed store (users/books/ratings/shelves) + recommend API
    load_ratings         -- read a ratings CSV into a DataFrame
    generate_synthetic   -- build a synthetic ratings dataset with latent taste clusters
"""

from .matrix import RatingsMatrix, load_ratings
from .recommender import Recommender
from .platform import Platform
from .data import generate_synthetic

__all__ = [
    "RatingsMatrix",
    "load_ratings",
    "Recommender",
    "Platform",
    "generate_synthetic",
]

__version__ = "1.0.0"
