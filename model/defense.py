"""Privacy defense experiment: before releasing the database, replace each
rating with a random star with probability q (randomized response).
Measure (a) how often the attacker still re-identifies a reader from 5
noisy ratings, and (b) how much a recommender trained on the released
data loses in accuracy (RMSE on held-out true ratings).
"""
import json
import time
import numpy as np
import pandas as pd
from core import Data, make_query, score_learned, score_rarity, top1_hit

rng = np.random.default_rng(11)
t0 = time.time()
r = pd.read_csv("ratings.csv").drop_duplicates(["user_id", "book_id"], keep="last")
U = r.user_id.values.astype(np.int32)
B = r.book_id.values.astype(np.int32)
R = r.rating.values.astype(np.int8)
n_users, n_books = int(U.max()), int(B.max())

# 5% of true ratings held out to score the recommender
test_mask = rng.random(len(R)) < 0.05
Utr, Btr, Rtr = U[~test_mask], B[~test_mask], R[~test_mask]
Ute, Bte, Rte = U[test_mask], B[test_mask], R[test_mask].astype(np.float64)

theta = np.array(json.load(open("results.json"))["theta"])


def make_data(u, b, rr):
    d = Data.__new__(Data)
    d.n_users, d.n_books = n_users, n_books
    d.load(u, b, rr)
    return d


truth = make_data(Utr, Btr, Rtr)  # what the attacker's outside knowledge comes from


def als(u, b, rr, f=20, iters=8, lam=5.0):
    y = rr.astype(np.float64)
    mu = y.mean()
    bu = np.zeros(n_users + 1)
    bi = np.zeros(n_books + 1)
    for _ in range(4):  # regularized biases
        bi = np.bincount(b, y - mu - bu[u], n_books + 1) / (np.bincount(b, minlength=n_books + 1) + 10)
        bu = np.bincount(u, y - mu - bi[b], n_users + 1) / (np.bincount(u, minlength=n_users + 1) + 10)
    res = y - mu - bu[u] - bi[b]
    P = rng.normal(0, 0.1, (n_users + 1, f))
    Q = rng.normal(0, 0.1, (n_books + 1, f))
    ou = np.argsort(u, kind="stable"); pu = np.concatenate([[0], np.cumsum(np.bincount(u, minlength=n_users + 1))])
    ob = np.argsort(b, kind="stable"); pb = np.concatenate([[0], np.cumsum(np.bincount(b, minlength=n_books + 1))])
    I = lam * np.eye(f)
    for _ in range(iters):
        for x in range(1, n_users + 1):
            idx = ou[pu[x]:pu[x + 1]]
            if len(idx):
                Qx = Q[b[idx]]
                P[x] = np.linalg.solve(Qx.T @ Qx + I, Qx.T @ res[idx])
        for x in range(1, n_books + 1):
            idx = ob[pb[x]:pb[x + 1]]
            if len(idx):
                Px = P[u[idx]]
                Q[x] = np.linalg.solve(Px.T @ Px + I, Px.T @ res[idx])
    return lambda uu, bb: np.clip(mu + bu[uu] + bi[bb] + (P[uu] * Q[bb]).sum(1), 1, 5)


def rmse(pred):
    return float(np.sqrt(np.mean((pred - Rte) ** 2)))


out = {"item_mean_rmse": None, "rows": []}
# simple reference point: predict each book's average rating
item_mean = np.bincount(Btr, Rtr.astype(float), n_books + 1) / np.maximum(np.bincount(Btr, minlength=n_books + 1), 1)
out["item_mean_rmse"] = rmse(item_mean[Bte])
print(f"item-mean baseline RMSE {out['item_mean_rmse']:.4f}")

N_Q = 500
for q in [0.0, 0.1, 0.25, 0.5, 0.75]:
    flip = rng.random(len(Rtr)) < q
    Rrel = Rtr.copy()
    Rrel[flip] = rng.integers(1, 6, flip.sum())
    released = make_data(Utr, Btr, Rrel)
    hit_l = hit_r = 0.0
    for _ in range(N_Q):
        tu, qb, qr = make_query(truth, rng, 5, 0.3)  # attacker knows true-ish ratings
        F = released.features(qb, qr)
        hit_l += top1_hit(score_learned(F, 5, theta), tu, rng)
        hit_r += top1_hit(score_rarity(F, 5), tu, rng)
    pred = als(Utr, Btr, Rrel)
    row = {"q": q, "reid_learned": round(100 * hit_l / N_Q, 1),
           "reid_rarity": round(100 * hit_r / N_Q, 1),
           "rec_rmse": round(rmse(pred(Ute, Bte)), 4)}
    out["rows"].append(row)
    print(row, f"({time.time()-t0:.0f}s)", flush=True)

json.dump(out, open("defense.json", "w"), indent=1)
print("saved defense.json")
