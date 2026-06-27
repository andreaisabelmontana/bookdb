"""BookDB demo: kindred readers, top-N recommendations, and eval metrics.

Run from the repo root::

    python demo.py

It loads the committed synthetic dataset (``data/ratings.csv``), spins up the
SQLite platform, prints a reader's kindred readers and their top recommendations
from both user-based and item-based collaborative filtering, then runs
leave-one-out evaluation comparing CF against the popularity baseline.
"""

from __future__ import annotations

import os

from bookdb.matrix import RatingsMatrix, load_ratings
from bookdb.platform import Platform
from bookdb.data import generate_synthetic
from bookdb.evaluate import compare_methods

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def _rule(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def main() -> None:
    # Load committed data; regenerate it in memory if data/ is absent.
    if os.path.exists(os.path.join(DATA, "ratings.csv")):
        df = load_ratings(os.path.join(DATA, "ratings.csv"))
        ds = generate_synthetic()  # for titles/metadata + the platform store
        platform = Platform.from_dataset(ds)
        source = "data/ratings.csv"
    else:
        ds = generate_synthetic()
        ds.ratings.to_csv(os.path.join(DATA, "ratings.csv"), index=False)
        df = ds.ratings
        platform = Platform.from_dataset(ds)
        source = "freshly generated synthetic dataset"

    matrix = RatingsMatrix.from_dataframe(df)
    print("BookDB demo")
    print("===========")
    print(f"dataset: {source}")
    print(
        f"{matrix.n_users} readers x {matrix.n_books} books, "
        f"{matrix.matrix.nnz} ratings "
        f"(density {matrix.matrix.nnz / (matrix.n_users * matrix.n_books):.1%})"
    )

    target = 0
    name = platform.user_name(target)
    cluster = platform.conn.execute(
        "SELECT cluster FROM users WHERE user_id=?", (target,)
    ).fetchone()["cluster"]
    _rule(f"Kindred readers for {name} (user {target}, taste cluster {cluster})")
    for uid, sim in platform.kindred_readers(target, k=5):
        their_cluster = platform.conn.execute(
            "SELECT cluster FROM users WHERE user_id=?", (uid,)
        ).fetchone()["cluster"]
        print(f"  user {uid:>3} ({platform.user_name(uid):<8})  "
              f"similarity={sim:.3f}  cluster={their_cluster}")

    _rule(f"Top-5 recommendations for {name} -- user-based CF")
    for title, score in platform.recommend_titles(target, n=5, method="user"):
        print(f"  {score:.2f}  {title}")

    _rule(f"Top-5 recommendations for {name} -- item-based CF")
    for title, score in platform.recommend_titles(target, n=5, method="item"):
        print(f"  {score:.2f}  {title}")

    _rule(f"Top-5 popularity baseline (non-personalized)")
    for title, score in platform.recommend_titles(target, n=5, method="popularity"):
        print(f"  {score:.2f}  {title}")

    _rule("Leave-one-out evaluation (held-out liked book, K=10)")
    results = compare_methods(matrix, k=10, min_rating=4.0, seed=0)
    for method in ("user", "item", "popularity"):
        print("  " + str(results[method]))

    best = max(("user", "item"), key=lambda m: results[m].recall)
    lift = results[best].recall - results["popularity"].recall
    print(
        f"\n  best CF ({best}) Recall@10 = {results[best].recall:.3f} vs "
        f"popularity {results['popularity'].recall:.3f}  "
        f"(+{lift:.3f}, "
        f"{results[best].recall / max(results['popularity'].recall, 1e-9):.1f}x)"
    )


if __name__ == "__main__":
    main()
