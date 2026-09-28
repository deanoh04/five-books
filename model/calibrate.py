"""Temperature-scale the learned scores so the displayed confidence matches
reality. Fit on training readers, check on held-out readers. Ranking is unchanged."""
import json
import numpy as np
from core import Data, make_query, score_learned

rng = np.random.default_rng(3)
data = Data()
perm = np.random.default_rng(7).permutation(np.arange(1, data.n_users + 1))
train_users = set(perm[: len(perm) // 2].tolist())
res = json.load(open("results.json"))
theta = np.array(res["theta"])


def collect(train, n):
    out = []
    while len(out) < n:
        k = int(rng.integers(1, 11)); noise = float(rng.choice([0.0, 0.3]))
        u, qb, qr = make_query(data, rng, k, noise)
        if (u in train_users) != train:
            continue
        s = score_learned(data.features(qb, qr), k, theta)
        fin = np.isfinite(s)
        out.append((s[fin] - s[u], s[u]))  # scores relative to truth
    return out


def nll(T, sets):
    return np.mean([np.log(np.exp(T * (d - d.max())).sum()) + T * d.max() for d, _ in sets])


tr = collect(True, 1500)
grid = np.linspace(0.5, 3, 51)
T = float(grid[np.argmin([nll(t, tr) for t in grid])])
print("temperature", T)

te = collect(False, 1500)
bins = np.zeros((10, 3))
for d, _ in te:
    p = np.exp(T * (d - d.max())); p /= p.sum()
    conf = p.max(); hit = float(d.max() == 0 and (d == 0).sum() == 1)
    b = min(int(conf * 10), 9); bins[b] += [1, conf, hit]
cal = [{"bucket": f"{i*10}-{i*10+10}%", "n": int(c), "avg_conf": round(100*a/c, 1), "hit_rate": round(100*h/c, 1)}
       for i, (c, a, h) in enumerate(bins) if c]
for c in cal: print(c)
ece = sum(c["n"] * abs(c["avg_conf"] - c["hit_rate"]) for c in cal) / sum(c["n"] for c in cal)
print(f"expected calibration error {ece:.1f} pts")
res["theta_uncalibrated"] = res.get("theta_uncalibrated", res["theta"])
res["theta"] = (np.array(res["theta_uncalibrated"]) * T).tolist()
res["temperature"] = T; res["calibration"] = cal; res["ece"] = round(ece, 1)
json.dump(res, open("results.json", "w"), indent=1)
