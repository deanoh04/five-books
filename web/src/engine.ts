// In-browser re-identification engine. Mirrors model/core.py exactly:
// same features, same learned weights, same tie handling.

export type Book = { id: number; title: string; author: string; year: number | null; image: string; support: number };
export type Clue = { book: number; rating: number; trueRating?: number };
export type Guess = { user: number; score: number; prob: number };
export type Result = {
  top: Guess[];
  candidates: number;      // readers who rated at least one clue book
  exactMatches: number;    // readers whose stars match every clue exactly (the naive baseline)
  truthRank?: number;      // rank of a known reader, for scoring
  truthProb?: number;
  topTied: number;         // readers tied for first place
};
export type Model = {
  theta: number[];
  n_users: number; n_books: number; n_ratings: number;
  book_shards: number; users_per_shard: number;
  curve: Record<string, { exact: number; rarity: number; learned: number; learned_top10: number }>;
  ks: number[];
  defense: { item_mean_rmse: number; rows: { q: number; reid_learned: number; reid_rarity: number; rec_rmse: number }[] };
  headline: Record<string, any> | null;
};

const BASE = "data";
// Binary shards are .bin; a host that only serves known file types can build with VITE_DATA_EXT=.wasm
const EXT: string = (import.meta as any).env?.VITE_DATA_EXT || ".bin";
let model: Model;
let shelves: Uint8Array;
export const books = new Map<number, Book>();

export async function loadMeta(): Promise<Model> {
  const [m, b, s] = await Promise.all([
    fetch(`${BASE}/model.json`).then(r => r.json()),
    fetch(`${BASE}/books.json`).then(r => r.json()),
    fetch(`${BASE}/shelves${EXT}`).then(r => r.arrayBuffer()),
  ]);
  model = m;
  for (const [id, title, author, year, image, support] of b as any[]) books.set(id, { id, title, author, year, image, support });
  shelves = new Uint8Array(s);
  return model;
}

type List = { users: Uint16Array; ratings: Uint8Array };
const bookShards = new Map<number, Promise<Map<number, List>>>();
const userShards = new Map<number, Promise<Map<number, { books: Uint16Array; ratings: Uint8Array }>>>();

function u16(buf: ArrayBuffer, off: number, n: number) { return new Uint16Array(buf.slice(off, off + 2 * n)); }

function bookShard(s: number) {
  if (!bookShards.has(s)) {
    bookShards.set(s, fetch(`${BASE}/books/${s}${EXT}`).then(r => r.arrayBuffer()).then(buf => {
      const dv = new DataView(buf); const out = new Map<number, List>(); let o = 0;
      while (o < buf.byteLength) {
        const id = dv.getUint16(o, true); const n = dv.getUint32(o + 2, true); o += 6;
        const users = u16(buf, o, n); o += 2 * n;
        const ratings = new Uint8Array(buf.slice(o, o + n)); o += n;
        out.set(id, { users, ratings });
      }
      return out;
    }));
  }
  return bookShards.get(s)!;
}

function userShard(s: number) {
  if (!userShards.has(s)) {
    userShards.set(s, fetch(`${BASE}/users/${s}${EXT}`).then(r => r.arrayBuffer()).then(buf => {
      const dv = new DataView(buf); const out = new Map<number, { books: Uint16Array; ratings: Uint8Array }>(); let o = 0;
      while (o < buf.byteLength) {
        const id = dv.getUint16(o, true); const n = dv.getUint16(o + 2, true); o += 4;
        const bk = u16(buf, o, n); o += 2 * n;
        const rt = new Uint8Array(buf.slice(o, o + n)); o += n;
        out.set(id, { books: bk, ratings: rt });
      }
      return out;
    }));
  }
  return userShards.get(s)!;
}

export async function getShelf(user: number) {
  const shard = await userShard(Math.floor((user - 1) / model.users_per_shard));
  const s = shard.get(user)!;
  return Array.from(s.books, (b, i) => ({ book: b, rating: s.ratings[i] }));
}

export async function identify(clues: Clue[], truth?: number): Promise<Result> {
  const n = model.n_users + 1;
  const t = model.theta;
  const score = new Float64Array(n);
  const matched = new Uint8Array(n);
  const exact = new Uint8Array(n);
  const lists = await Promise.all(clues.map(c => bookShard(c.book % model.book_shards).then(m => m.get(c.book)!)));
  clues.forEach((c, i) => {
    const L = lists[i];
    const w = 1 / Math.log(L.users.length + 1);
    for (let j = 0; j < L.users.length; j++) {
      const u = L.users[j];
      const d = Math.abs(L.ratings[j] - c.rating);
      score[u] += (d === 0 ? t[0] * w + t[4] : d === 1 ? t[1] * w : t[2] * w) + t[3];
      matched[u]++;
      if (d === 0) exact[u]++;
    }
  });
  const cand: number[] = [];
  let exactMatches = 0;
  for (let u = 1; u < n; u++) {
    if (!matched[u]) continue;
    score[u] += t[5] * Math.log1p(shelves[u]);
    cand.push(u);
    if (exact[u] >= clues.length) exactMatches++;
  }
  let max = -Infinity;
  for (const u of cand) if (score[u] > max) max = score[u];
  let z = 0;
  for (const u of cand) z += Math.exp(score[u] - max);
  cand.sort((a, b) => score[b] - score[a]);
  const top = cand.slice(0, 10).map(u => ({ user: u, score: score[u], prob: Math.exp(score[u] - max) / z }));
  let topTied = 0;
  for (const u of cand) { if (score[u] === max) topTied++; else break; }
  const res: Result = { top, candidates: cand.length, exactMatches, topTied };
  if (truth !== undefined) {
    let r = 1;
    for (const u of cand) if (score[u] > score[truth]) r++;
    res.truthRank = matched[truth] ? r : Infinity;
    res.truthProb = matched[truth] ? Math.exp(score[truth] - max) / z : 0;
  }
  return res;
}

export function getModel() { return model; }
