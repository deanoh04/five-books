"""Train the learned ranker and evaluate all three re-identification methods.

Train/test split is by reader: training queries come from one half of the
readers, every reported number comes from the other half.
"""
import json
import time
import numpy as np
from scipy.optimize import minimize
from core import (Data, make_query, score_exact, score_rarity, score_learned,
                  top1_hit, rank_of, N_FEATURES, FEATURE_NAMES)

rng = np.random.default_rng(7)
t0 = time.time()
data = Data()
print(f"loaded {data.n_users} readers, {data.n_books} books, "
      f"{len(data.b_users)} ratings in {time.time()-t0:.1f}s")

# ---------------------------------------------------------------- split
users = np.arange(1, data.n_users + 1)
perm = rng.permutation(users)
train_users = set(perm[: len(perm) // 2].tolist())


def query_from(pool_is_train, k, noise):
    while True:
        u, qb, qr = make_query(data, rng, k, noise)
        if (u in train_users) == pool_is_train:
            return u, qb, qr


# ---------------------------------------------------------------- training
HARD = 400
train_sets = []
for i in range(3000):
    k = int(rng.integers(1, 9))
    noise = float(rng.choice([0.0, 0.2, 0.4]))
    u, qb, qr = query_from(True, k, noise)
    F = data.features(qb, qr)
    s = score_rarity(F, k)
    cand = np.flatnonzero(np.isfinite(s))
    if len(cand) > HARD:
        # hardest negatives: the readers the heuristic already confuses with the target
        top = cand[np.argpartition(-s[cand], HARD)[:HARD]]
        cand = np.union1d(top, [u])
    X = F[cand]
    y = int(np.flatnonzero(cand == u)[0])
    train_sets.append((X, y))
print(f"built {len(train_sets)} training queries in {time.time()-t0:.1f}s")

# standardize features so the L2 penalty treats them evenly
allX = np.vstack([X for X, _ in train_sets])
mu, sd = allX.mean(0), allX.std(0) + 1e-9


def loss_grad(theta):
    L, G = 0.0, np.zeros_like(theta)
    for X, y in train_sets:
        Z = (X - mu) / sd
        s = Z @ theta
        m = s.max()
        p = np.exp(s - m)
        p /= p.sum()
        L += -(s[y] - m) + np.log(np.exp(s - m).sum())
        G += Z.T @ p - Z[y]
    lam = 1e-2
    n = len(train_sets)
    return L / n + lam * theta @ theta, G / n + 2 * lam * theta


res = minimize(loss_grad, np.zeros(N_FEATURES), jac=True, method="L-BFGS-B")
theta_std = res.x
# fold standardization back in: s = ((F - mu)/sd) @ t = F @ (t/sd) + const
theta = theta_std / sd
print("learned weights (raw feature scale):")
for n, t in zip(FEATURE_NAMES, theta):
    print(f"  {n:40s} {t:+.3f}")

# ---------------------------------------------------------------- evaluation
KS = [1, 2, 3, 4, 5, 6, 8, 10]
NOISES = [0.0, 0.3]
N_TEST = 600
results = {}
for noise in NOISES:
    for k in KS:
        hits = {"exact": 0.0, "rarity": 0.0, "learned": 0.0}
        ranks = []
        for _ in range(N_TEST):
            u, qb, qr = query_from(False, k, noise)
            F = data.features(qb, qr)
            hits["exact"] += top1_hit(score_exact(F, k), u, rng)
            hits["rarity"] += top1_hit(score_rarity(F, k), u, rng)
            sl = score_learned(F, k, theta)
            hits["learned"] += top1_hit(sl, u, rng)
            ranks.append(rank_of(sl, u))
        row = {m: round(100 * h / N_TEST, 1) for m, h in hits.items()}
        row["learned_top10"] = round(100 * np.mean(np.array(ranks) <= 10), 1)
        results[f"{noise}|{k}"] = row
        print(f"noise={noise:.1f} k={k:2d}  exact={row['exact']:5.1f}%  "
              f"rarity={row['rarity']:5.1f}%  learned={row['learned']:5.1f}%  "
              f"learned top10={row['learned_top10']:5.1f}%   ({time.time()-t0:.0f}s)")

json.dump({"theta": theta.tolist(), "feature_names": FEATURE_NAMES,
           "results": results, "n_test": N_TEST, "ks": KS, "noises": NOISES},
          open("results.json", "w"), indent=1)
print("saved results.json")
