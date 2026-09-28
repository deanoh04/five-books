"""Export the database in small binary shards so the browser only downloads
what one search needs (a few hundred KB), plus book metadata and model weights.

Formats (all little-endian):
  books/<s>.bin   shard s holds every book with book_id % 128 == s
                  repeated: book_id u16, count u32, users u16[count], ratings u8[count]
  users/<s>.bin   shard s holds readers (s*500+1 .. s*500+500)
                  repeated: user_id u16, count u16, books u16[count], ratings u8[count]
  shelves.bin     u8[n_users+1]: shelf size of every reader (for the shelf-size feature)
  books.json      [id, title, author, year, image, support] per book
  model.json      learned weights + evaluation results for the charts
"""
import json
import os
import numpy as np
import pandas as pd
from core import Data

OUT = "../web/public/data"
BOOK_SHARDS = 128
USERS_PER_SHARD = 500
os.makedirs(f"{OUT}/books", exist_ok=True)
os.makedirs(f"{OUT}/users", exist_ok=True)

d = Data()
assert d.n_users < 65536 and d.n_books < 65536

for s in range(BOOK_SHARDS):
    parts = []
    for b in range(s if s else BOOK_SHARDS, d.n_books + 1, BOOK_SHARDS):
        lo, hi = d.b_ptr[b], d.b_ptr[b + 1]
        n = hi - lo
        parts += [np.array([b], "<u2").tobytes(), np.array([n], "<u4").tobytes(),
                  d.b_users[lo:hi].astype("<u2").tobytes(), d.b_ratings[lo:hi].astype("u1").tobytes()]
    open(f"{OUT}/books/{s}.bin", "wb").write(b"".join(parts))

n_user_shards = (d.n_users + USERS_PER_SHARD - 1) // USERS_PER_SHARD
for s in range(n_user_shards):
    parts = []
    for u in range(s * USERS_PER_SHARD + 1, min((s + 1) * USERS_PER_SHARD, d.n_users) + 1):
        lo, hi = d.u_ptr[u], d.u_ptr[u + 1]
        parts += [np.array([u, hi - lo], "<u2").tobytes(),
                  d.u_books[lo:hi].astype("<u2").tobytes(), d.u_ratings[lo:hi].astype("u1").tobytes()]
    open(f"{OUT}/users/{s}.bin", "wb").write(b"".join(parts))

assert d.shelf.max() < 256
open(f"{OUT}/shelves.bin", "wb").write(d.shelf.astype("u1").tobytes())

books = pd.read_csv("books.csv")
rows = []
for _, r in books.iterrows():
    img = r.image_url if isinstance(r.image_url, str) and "nophoto" not in r.image_url else ""
    year = int(r.original_publication_year) if pd.notna(r.original_publication_year) else None
    author = str(r.authors).split(",")[0]
    rows.append([int(r.book_id), str(r.title), author, year, img, int(d.book_support[r.book_id])])
json.dump(rows, open(f"{OUT}/books.json", "w"), separators=(",", ":"), ensure_ascii=False)

res = json.load(open("results.json"))
model = {
    "theta": res["theta"],
    "n_users": d.n_users, "n_books": d.n_books, "n_ratings": int(len(d.b_users)),
    "book_shards": BOOK_SHARDS, "users_per_shard": USERS_PER_SHARD,
    "curve": res["results"], "ks": res["ks"],
    "defense": json.load(open("defense.json")),
    "headline": json.load(open("headline.json")) if os.path.exists("headline.json") else None,
}
json.dump(model, open(f"{OUT}/model.json", "w"), indent=1)

tot = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(OUT) for f in fs)
print(f"exported {BOOK_SHARDS} book shards, {n_user_shards} user shards, total {tot/1e6:.1f} MB")
print("largest book shard", max(os.path.getsize(f"{OUT}/books/{s}.bin") for s in range(BOOK_SHARDS)) / 1e3, "KB")
