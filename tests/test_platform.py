"""Tests for the SQLite platform layer and its recommender wiring."""

import pytest

from bookdb.platform import Platform
from bookdb.data import generate_synthetic


@pytest.fixture
def platform():
    ds = generate_synthetic(seed=5)
    p = Platform.from_dataset(ds)
    yield p
    p.close()


def test_from_dataset_loads_all_rows():
    ds = generate_synthetic(seed=5)
    with Platform.from_dataset(ds) as p:
        n_ratings = p.conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
        n_users = p.conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        n_books = p.conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    assert n_ratings == len(ds.ratings)
    assert n_users == len(ds.users)
    assert n_books == len(ds.books)


def test_add_rating_invalidates_and_updates_recommender():
    with Platform() as p:
        p.add_user(1, "Reader One")
        p.add_user(2, "Reader Two")
        for b in range(5):
            p.add_book(b, f"Book {b}")
        # two users with identical taste
        for b in range(5):
            p.add_rating(1, b, 5.0 if b < 3 else 1.0)
            p.add_rating(2, b, 5.0 if b < 3 else 1.0)
        kindred = p.kindred_readers(1, k=1)
        assert kindred[0][0] == 2


def test_recommend_for_excludes_read(platform):
    user = 0
    read = set(platform.recommender().ratings.rated_book_ids(user))
    recs = platform.recommend_for(user, n=10, method="item")
    assert {b for b, _ in recs}.isdisjoint(read)


def test_recommend_titles_returns_strings(platform):
    recs = platform.recommend_titles(0, n=3, method="user")
    assert len(recs) == 3
    assert all(isinstance(title, str) for title, _ in recs)


def test_shelve_and_read_back():
    with Platform() as p:
        p.add_user(1, "Reader")
        p.add_book(7, "Some Book")
        p.shelve(1, 7, "to-read")
        assert p.shelf(1, "to-read") == [7]
        assert p.shelf(1, "read") == []


def test_kindred_readers_from_platform(platform):
    kindred = platform.kindred_readers(0, k=5)
    assert len(kindred) <= 5
    assert all(sim > 0 for _, sim in kindred)
