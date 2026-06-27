"""Tests for the ratings-matrix construction and id<->index maps."""

import numpy as np
import pandas as pd
import pytest

from bookdb.matrix import RatingsMatrix, load_ratings


def test_from_dataframe_builds_correct_shape_and_values():
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "book_id": ["b1", "b2", "b1"],
            "rating": [5.0, 3.0, 4.0],
        }
    )
    m = RatingsMatrix.from_dataframe(df)
    assert m.n_users == 2
    assert m.n_books == 2
    assert m.matrix.nnz == 3
    # u1's rating of b1
    i = m.user_index["u1"]
    j = m.book_index["b1"]
    assert m.matrix[i, j] == 5.0


def test_duplicate_pairs_are_averaged():
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "book_id": ["b1", "b1"],
            "rating": [2.0, 4.0],
        }
    )
    m = RatingsMatrix.from_dataframe(df)
    assert m.matrix.nnz == 1
    assert m.matrix[m.user_index["u1"], m.book_index["b1"]] == 3.0


def test_id_index_maps_are_inverse():
    df = pd.DataFrame(
        {"user_id": [10, 20, 30], "book_id": [1, 2, 3], "rating": [5, 5, 5]}
    )
    m = RatingsMatrix.from_dataframe(df)
    for uid in (10, 20, 30):
        assert m.user_ids[m.user_index[uid]] == uid


def test_rated_book_ids():
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "book_id": ["b1", "b3", "b2"],
            "rating": [5.0, 4.0, 3.0],
        }
    )
    m = RatingsMatrix.from_dataframe(df)
    assert set(m.rated_book_ids("u1")) == {"b1", "b3"}
    assert set(m.rated_book_ids("u2")) == {"b2"}


def test_empty_frame_raises():
    with pytest.raises(ValueError):
        RatingsMatrix.from_dataframe(pd.DataFrame(columns=["user_id", "book_id", "rating"]))


def test_load_ratings_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("user_id,rating\n1,5\n")
    with pytest.raises(ValueError):
        load_ratings(str(p))


def test_load_ratings_roundtrip(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("user_id,book_id,rating\n1,2,5\n1,3,4\n")
    df = load_ratings(str(p))
    assert list(df.columns) == ["user_id", "book_id", "rating"]
    assert len(df) == 2
    assert df["rating"].dtype == float
