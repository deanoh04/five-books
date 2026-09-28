"""Shared data loading and scoring for the reading-fingerprint re-identification model.

Threat model (same as Narayanan & Shmatikov's Netflix study):
the attacker holds the full "anonymized" ratings database and knows a few
(book, approximate star rating) facts about one target. Goal: pick the target
out of every reader in the database.
"""
import numpy as np
import pandas as pd

N_FEATURES = 6
FEATURE_NAMES = [
    "rarity-weighted exact matches",
    "rarity-weighted off-by-one matches",
    "rarity-weighted big mismatches",
    "books matched (count)",
    "exact matches (count)",
    "log shelf size",
]


class Data:
    def __init__(self, ratings_path="ratings.csv"):
        r = pd.read_csv(ratings_path)
        r = r.drop_duplicates(["user_id", "book_id"], keep="last")
        self.n_users = int(r.user_id.max())
        self.n_books = int(r.book_id.max())
        self.load(r.user_id.values.astype(np.int32),
                  r.book_id.values.astype(np.int32),
                  r.rating.values.astype(np.int8))

    def load(self, users, books, ratings):
        # CSR by book: users sorted inside each book
        order = np.lexsort((users, books))
        self.b_users = users[order]
        self.b_ratings = ratings[order]
        counts = np.bincount(books, minlength=self.n_books + 1)
        self.b_ptr = np.concatenate([[0], np.cumsum(counts)])
        # CSR by user
        order_u = np.lexsort((books, users))
        self.u_books = books[order_u]
        self.u_ratings = ratings[order_u]
        ucounts = np.bincount(users, minlength=self.n_users + 1)
        self.u_ptr = np.concatenate([[0], np.cumsum(ucounts)])
        self.book_support = counts.astype(np.float64)
        self.shelf = ucounts.astype(np.float64)
        # rarity weight: rare books are far more identifying (1 / log support)
        self.w = np.zeros(self.n_books + 1)
        nz = counts > 0
        self.w[nz] = 1.0 / np.log(counts[nz] + 1.0)
        self.log_shelf = np.log1p(self.shelf)

    def shelf_of(self, u):
        s, e = self.u_ptr[u], self.u_ptr[u + 1]
        return self.u_books[s:e], self.u_ratings[s:e]

    def features(self, q_books, q_ratings):
        """Dense feature matrix (n_users+1, N_FEATURES) for one query."""
        n = self.n_users + 1
        F = np.zeros((n, N_FEATURES))
        for b, rq in zip(q_books, q_ratings):
            s, e = self.b_ptr[b], self.b_ptr[b + 1]
            us, rs = self.b_users[s:e], self.b_ratings[s:e]
            d = np.abs(rs.astype(np.int16) - int(rq))
            wb = self.w[b]
            F[us, 0] += wb * (d == 0)
            F[us, 1] += wb * (d == 1)
            F[us, 2] += wb * (d >= 2)
            F[us, 3] += 1
            F[us, 4] += (d == 0)
        F[:, 5] = self.log_shelf
        return F


def make_query(data, rng, k, noise):
    """Pick a random reader, reveal k of their books, jitter each rating by
    +/-1 star with probability `noise` (imperfect memory)."""
    while True:
        u = int(rng.integers(1, data.n_users + 1))
        books, ratings = data.shelf_of(u)
        if len(books) >= k:
            break
    idx = rng.choice(len(books), size=k, replace=False)
    qb = books[idx]
    qr = ratings[idx].astype(np.int16).copy()
    flip = rng.random(k) < noise
    step = rng.choice([-1, 1], size=k)
    qr[flip] = np.clip(qr[flip] + step[flip], 1, 5)
    # if clipping undid the change, push the other way
    same = flip & (qr == ratings[idx])
    qr[same] = np.clip(qr[same] - step[same], 1, 5)
    return u, qb, qr


# ---------------------------------------------------------------- scorers
def score_exact(F, k):
    # baseline: readers whose ratings match every revealed book exactly
    return np.where(F[:, 4] >= k, 1.0, -np.inf)


def score_rarity(F, k):
    # hand-tuned Netflix-study style score: rarity-weighted, +/-1 star tolerance
    s = F[:, 0] + 0.5 * F[:, 1] - 1.0 * F[:, 2]
    return np.where(F[:, 3] > 0, s, -np.inf)


def score_learned(F, k, theta):
    s = F @ theta
    return np.where(F[:, 3] > 0, s, -np.inf)


def top1_hit(scores, truth, rng):
    """1 if the truth is the top-scoring reader. Ties are broken at random,
    so a tie among m readers counts as 1/m in expectation."""
    best = scores.max()
    if not np.isfinite(best):
        return 0.0
    tied = np.flatnonzero(scores == best)
    return 1.0 / len(tied) if truth in tied else 0.0


def rank_of(scores, truth):
    return int((scores > scores[truth]).sum()) + 1
