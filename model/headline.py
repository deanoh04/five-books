"""Tighter numbers for the headline claims (3,000 test readers each),
a harder attacker scenario, and a calibration check of the confidence score."""
import json
import numpy as np
from core import Data, make_query, score_exact, score_rarity, score_learned, top1_hit

rng = np.random.default_rng(7)
data = Data()
perm = rng.permutation(np.arange(1, data.n_users + 1))
train_users = set(perm[: len(perm) // 2].tolist())  # same split as train_eval.py
theta = np.array(json.load(open("results.json"))["theta"])
N = 3000


def test_query(k, noise, popular=False):
    while True:
        u, qb, qr = make_query(data, rng, k, noise)
        if u in train_users:
            continue
        if popular:
            # harder: attacker only knows the target's k MOST popular books
            books, ratings = data.shelf_of(u)
            idx = np.argsort(-data.book_support[books])[:k]
            qb, qr = books[idx], ratings[idx].astype(np.int16).copy()
            flip = rng.random(k) < noise
            step = rng.choice([-1, 1], size=k)
            new = np.clip(qr + step, 1, 5)
            new[new == qr] = np.clip(qr - step, 1, 5)[new == qr]
            qr = np.where(flip, new, qr)
        return u, qb, qr


out = {}
conf_bins = np.zeros((10, 3))  # count, sum confidence, sum hits
for name, k, noise, popular in [("k5_noise30", 5, 0.3, False), ("k5_clean", 5, 0.0, False),
                                ("k3_noise30", 3, 0.3, False), ("k3_clean", 3, 0.0, False),
                                ("k5_noise30_popular", 5, 0.3, True),
                                ("k8_noise30_popular", 8, 0.3, True)]:
    h = {"exact": 0.0, "rarity": 0.0, "learned": 0.0}
    for _ in range(N):
        u, qb, qr = test_query(k, noise, popular)
        F = data.features(qb, qr)
        h["exact"] += top1_hit(score_exact(F, k), u, rng)
        h["rarity"] += top1_hit(score_rarity(F, k), u, rng)
        s = score_learned(F, k, theta)
        hit = top1_hit(s, u, rng)
        h["learned"] += hit
        fin = s[np.isfinite(s)]
        p = np.exp(fin - fin.max())
        conf = 1.0 / p.sum()  # softmax probability of the top guess
        b = min(int(conf * 10), 9)
        conf_bins[b] += [1, conf, hit]
    row = {m: round(100 * v / N, 1) for m, v in h.items()}
    se = round(100 * np.sqrt((row["learned"] / 100) * (1 - row["learned"] / 100) / N), 1)
    row["learned_se"] = se
    out[name] = row
    print(name, row, flush=True)

# calibration: in each confidence bucket, average confidence vs actual hit rate
cal = [{"bucket": f"{i*10}-{i*10+10}%", "n": int(c), "avg_conf": round(100*sc/c, 1), "hit_rate": round(100*sh/c, 1)}
       for i, (c, sc, sh) in enumerate(conf_bins) if c > 0]
for c in cal: print(c)
out["calibration"] = cal
json.dump(out, open("headline.json", "w"), indent=1)
print("saved headline.json")
