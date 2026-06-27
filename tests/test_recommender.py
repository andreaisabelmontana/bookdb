"""Tests for the collaborative-filtering engine.

The small fixtures plant explicit structure -- a kindred reader, a block of
co-loved books -- so each assertion checks a specific CF behaviour rather than
an aggregate. The larger synthetic-data test confirms CF beats the popularity
baseline on planted-cluster data.
"""

import numpy as np
import pandas as pd
import pytest

from bookdb.matrix import RatingsMatrix
from bookdb.recommender import Recommender
from bookdb.data import generate_synthetic
from bookdb.evaluate import leave_one_out


def _matrix(rows):
    """rows: list of (user_id, book_id, rating)."""
    df = pd.DataFrame(rows, columns=["user_id", "book_id", "rating"])
    return RatingsMatrix.from_dataframe(df)


# --------------------------------------------------------------------------
# user-based CF: the planted kindred reader is found
# --------------------------------------------------------------------------

def test_user_based_finds_planted_kindred_reader():
    # Users "alice" and "twin" rate the same books almost identically.
    # "opposite" rates the same books the other way around.
    rows = []
    for b in range(6):
        rows.append(("alice", b, 5.0 if b < 3 else 1.0))
        rows.append(("twin", b, 5.0 if b < 3 else 1.0))   # mirror of alice
        rows.append(("opposite", b, 1.0 if b < 3 else 5.0))  # anti-correlated
    # a fourth, unrelated user adds noise
    for b in range(6):
        rows.append(("noise", b, 3.0))
    m = _matrix(rows)
    rec = Recommender(m)

    kindred = rec.kindred_readers("alice", k=3)
    kindred_ids = [uid for uid, _ in kindred]

    assert kindred_ids[0] == "twin", f"expected twin as #1 kindred, got {kindred}"
    # the anti-correlated user must not be a (positive-similarity) kindred reader
    assert "opposite" not in kindred_ids


def test_kindred_similarity_is_positive_and_sorted():
    rows = []
    for b in range(8):
        rows.append(("u1", b, 5.0 if b % 2 == 0 else 2.0))
        rows.append(("u2", b, 5.0 if b % 2 == 0 else 2.0))
        rows.append(("u3", b, 4.5 if b % 2 == 0 else 2.5))
        rows.append(("u4", b, 2.0 if b % 2 == 0 else 5.0))
    rec = Recommender(_matrix(rows))
    kindred = rec.kindred_readers("u1", k=3)
    sims = [s for _, s in kindred]
    assert all(s > 0 for s in sims)               # only positive taste matches
    assert sims == sorted(sims, reverse=True)     # descending order


# --------------------------------------------------------------------------
# item-based CF: co-loved books rank together
# --------------------------------------------------------------------------

def test_item_based_ranks_coloved_books_together():
    # Books 0,1,2 are "co-loved": the same readers rate them all highly.
    # Books 3,4,5 are a different co-loved block.
    rows = []
    fans_a = ["a1", "a2", "a3", "a4"]
    fans_b = ["b1", "b2", "b3", "b4"]
    for u in fans_a:
        for b in (0, 1, 2):
            rows.append((u, b, 5.0))
        for b in (3, 4, 5):
            rows.append((u, b, 1.0))
    for u in fans_b:
        for b in (0, 1, 2):
            rows.append((u, b, 1.0))
        for b in (3, 4, 5):
            rows.append((u, b, 5.0))
    rec = Recommender(_matrix(rows))

    # Books most similar to book 0 should be its co-loved block (1 and 2),
    # ranked above the other block (3,4,5).
    neighbours = rec.similar_books(0, n=5)
    neighbour_ids = [bid for bid, _ in neighbours]
    top_two = set(neighbour_ids[:2])
    assert top_two == {1, 2}, f"co-loved block not on top: {neighbours}"


def test_item_based_recommends_from_the_users_loved_block():
    rows = []
    # target user loves block {0,1,2}; we hide book 2 so it can be recommended
    rows += [("target", 0, 5.0), ("target", 1, 5.0)]
    # community establishes that {0,1,2} are co-loved and {3,4,5} are separate
    for u in ["c1", "c2", "c3"]:
        rows += [(u, 0, 5.0), (u, 1, 5.0), (u, 2, 5.0),
                 (u, 3, 1.0), (u, 4, 1.0), (u, 5, 1.0)]
    for u in ["d1", "d2", "d3"]:
        rows += [(u, 0, 1.0), (u, 1, 1.0), (u, 2, 1.0),
                 (u, 3, 5.0), (u, 4, 5.0), (u, 5, 5.0)]
    rec = Recommender(_matrix(rows))
    recs = rec.recommend_for("target", n=1, method="item")
    assert recs[0][0] == 2, f"expected co-loved book 2 first, got {recs}"


# --------------------------------------------------------------------------
# recommendations exclude already-read books
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["user", "item", "popularity"])
def test_recommendations_exclude_already_read(method):
    ds = generate_synthetic(seed=3)
    m = RatingsMatrix.from_dataframe(ds.ratings)
    rec = Recommender(m)
    user = ds.ratings["user_id"].iloc[0]
    read = set(m.rated_book_ids(user))
    recs = rec.recommend_for(user, n=20, method=method)
    rec_ids = {bid for bid, _ in recs}
    assert rec_ids.isdisjoint(read), (
        f"{method} recommended already-read books: {rec_ids & read}"
    )


def test_recommendations_are_sorted_descending():
    ds = generate_synthetic(seed=3)
    rec = Recommender(RatingsMatrix.from_dataframe(ds.ratings))
    scores = [s for _, s in rec.recommend_for(0, n=10, method="user")]
    assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------
# CF beats the popularity baseline on planted-cluster data
# --------------------------------------------------------------------------

def test_cf_beats_popularity_baseline_recall_at_k():
    ds = generate_synthetic(seed=7)
    m = RatingsMatrix.from_dataframe(ds.ratings)

    pop = leave_one_out(m, method="popularity", k=10, seed=0)
    user = leave_one_out(m, method="user", k=10, seed=0)
    item = leave_one_out(m, method="item", k=10, seed=0)

    # both personalized methods must clearly outrank the non-personalized one
    assert user.recall > pop.recall, (user.recall, pop.recall)
    assert item.recall > pop.recall, (item.recall, pop.recall)
    # and the lift should be substantial, not marginal, on planted structure
    assert max(user.recall, item.recall) >= 2 * pop.recall


def test_drop_rating_preserves_shape_and_removes_entry():
    ds = generate_synthetic(seed=1)
    m = RatingsMatrix.from_dataframe(ds.ratings)
    user = ds.ratings["user_id"].iloc[0]
    book = m.rated_book_ids(user)[0]
    reduced = m.drop_rating(user, book)
    assert reduced.matrix.shape == m.matrix.shape
    assert book not in reduced.rated_book_ids(user)
    assert reduced.matrix.nnz == m.matrix.nnz - 1


def test_cold_start_user_falls_back_to_popularity():
    ds = generate_synthetic(seed=2)
    m = RatingsMatrix.from_dataframe(ds.ratings)
    rec = Recommender(m)
    # an unknown user id should not crash and should return popular books
    recs = rec.recommend_for("nobody-unknown", n=5, method="user")
    pop = rec.recommend_for("nobody-unknown", n=5, method="popularity")
    assert [b for b, _ in recs] == [b for b, _ in pop]
