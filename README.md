# BookDB

A book recommendation engine wrapped in a small social reading platform: shelve
and rate what you've read, find readers with taste like yours ("kindred
readers"), and get pointed at your next book from what they loved.

The engine is neighborhood **collaborative filtering** — user-based and
item-based, with a popularity baseline to beat. The platform is a SQLite store
(users / books / ratings / shelves) with a clean Python API that wires the data
into the recommender.

- **Live page:** https://andreaisabelmontana.github.io/bookdb/

## How it works

### Ratings matrix
Ratings are a long-form `(user_id, book_id, rating)` table. `RatingsMatrix`
turns them into a sparse `users x books` CSR matrix and keeps the id↔index maps
so the rest of the code works in index space and translates back at the edges.
The committed dataset is **200 readers × 120 books, 3,496 ratings (14.6%
density)** — sparse, like the real thing.

### User-based CF — kindred readers
Similarity between two readers is the **cosine** of their *mean-centered* rating
vectors (centering removes the harsh-vs-generous-rater bias so similarity
reflects taste, not scale). A reader's *kindred readers* are their top-k
positive-similarity neighbours. A book's predicted score is the
similarity-weighted average of how those neighbours rated it, added back onto
the reader's own mean:

```
score(u, i) = mu_u + ( Σ_{v∈N(u)} sim(u,v)·(r_{v,i} − mu_v) ) / Σ_{v∈N(u)} |sim(u,v)|
```

`N(u)` is the top-k similar users who rated book `i`. Books supported by fewer
than `min_support` neighbours are dropped so a single noisy neighbour can't
float a book to the top.

### Item-based CF — co-loved books
Symmetric, on the book columns, using **adjusted cosine** (columns mean-centered
per book). Centering lets oppositely-loved books take a *negative* similarity
instead of an artificially positive one. A book scores high for a reader when it
is similar to the books they already rated highly:

```
score(u, i) = ( Σ_{j∈rated(u)} sim(i,j)·r_{u,j} ) / Σ_{j∈rated(u)} |sim(i,j)|
```

Co-loved books — rated together by the same readers — get high mutual similarity
and are recommended together.

### Popularity baseline
Non-personalized: a count-damped mean rating, `mean·n/(n+C)`, which pulls
thinly-rated books toward the global mean. This is the bar CF has to clear.

In every method, **already-read books are excluded** from the returned list, and
an unknown (cold-start) user falls back to popularity.

## The platform layer

`Platform` is a SQLite store with four tables (`users`, `books`, `ratings`,
`shelves`) and a Python API on top:

```python
from bookdb import Platform, generate_synthetic

platform = Platform.from_dataset(generate_synthetic())

platform.kindred_readers(user_id=0, k=5)        # nearest readers by taste
platform.recommend_for(0, n=10, method="user")  # top-N (book_id, score)
platform.recommend_titles(0, n=10, method="item")
platform.add_rating(0, book_id=42, rating=5.0)  # write -> recommender refreshes
platform.shelve(0, 42, shelf="favourites")
```

The recommender is rebuilt lazily whenever a rating changes, so reads always
reflect the latest writes.

## Evaluation

Leave-one-out ranking eval (`bookdb/evaluate.py`): for each reader hold out one
of their highly-rated books, rebuild the recommender without it, ask for a
top-K list, and check whether the held-out book comes back. Reported as
**HitRate@K** and **Recall@K**, averaged over readers.

Real numbers on the committed synthetic dataset (planted taste clusters,
`seed=0`, held-out books rated ≥ 4.0, n = 200 readers):

| Recall@K | user-based | item-based | popularity |
|---------:|:----------:|:----------:|:----------:|
| **@5**   | 0.135      | 0.215      | 0.025      |
| **@10**  | 0.400      | 0.560      | 0.095      |
| **@20**  | 0.740      | 0.940      | 0.135      |

At K=10, item-based CF recovers the held-out book **5.9×** as often as the
popularity baseline (0.560 vs 0.095); user-based is **4.2×** (0.400). Both
personalized methods clearly beat the non-personalized baseline at every K — the
planted structure (readers cluster by taste; books cluster by genre) is exactly
what neighborhood CF is built to recover.

## Data

`data/ratings.csv`, `data/books.csv`, `data/users.csv` — a **synthetic** ratings
dataset with latent taste clusters, generated deterministically by
`bookdb/data.py` (`generate_synthetic`, `seed=7`). Each reader belongs to a taste
cluster and rates books in their cluster's genre highly (with Gaussian noise),
so kindred readers and co-loved books actually exist to be found. Clearly
synthetic and reproducible — regenerate with `python -c "from bookdb.data import
generate_synthetic; generate_synthetic()"`.

## Run it

```bash
pip install -r requirements.txt
python demo.py          # kindred readers + top-N recs + eval metrics
python -m pytest -q     # 24 tests
```

`demo.py` prints a reader's kindred readers, their top recommendations from both
CF methods and the popularity baseline, and the leave-one-out eval table above.

## Tests

`python -m pytest -q` → **24 passed**. They check the behaviours that matter:

- user-based CF finds the planted kindred reader (and rejects the anti-correlated one);
- item-based CF ranks co-loved books together and recommends from the user's loved block;
- recommendations exclude already-read books (all three methods);
- Recall@K beats the popularity baseline on planted-cluster data;
- matrix construction, id↔index maps, cold-start fallback, and the SQLite platform layer.

## Layout

```
bookdb/
  matrix.py        sparse user×book matrix + id↔index maps
  recommender.py   user-based & item-based CF + popularity baseline
  data.py          synthetic clustered ratings generator
  evaluate.py      leave-one-out Recall@K / HitRate@K
  platform.py      SQLite store + recommend API
data/              committed synthetic dataset (ratings/books/users)
tests/             pytest suite (24 tests)
demo.py            end-to-end demo
```

## License

MIT — see [LICENSE](LICENSE).
