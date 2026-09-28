import { useEffect, useState } from "react";
import { Result, books, getModel, getShelf, identify } from "./engine";
import { Cover, Stars, shortTitle } from "./Cover";

const fmt = (n: number) => n.toLocaleString("en-US");
type Shelf = { book: number; rating: number }[];

export default function Hide() {
  const m = getModel();
  const [reader, setReader] = useState<number | null>(null);
  const [shelf, setShelf] = useState<Shelf>([]);
  const [picked, setPicked] = useState<number[]>([]);
  const [result, setResult] = useState<Result | null>(null);
  const [sort, setSort] = useState<"popular" | "rare">("popular");

  async function draw() {
    const u = 1 + Math.floor(Math.random() * m.n_users);
    setReader(u); setShelf(await getShelf(u)); setPicked([]); setResult(null);
  }

  useEffect(() => {
    if (reader === null || picked.length === 0) { setResult(null); return; }
    let live = true;
    const clues = picked.map(b => ({ book: b, rating: shelf.find(s => s.book === b)!.rating }));
    identify(clues, reader).then(r => { if (live) setResult(r); });
    return () => { live = false; };
  }, [picked, reader, shelf]);

  const toggle = (b: number) =>
    setPicked(p => p.includes(b) ? p.filter(x => x !== b) : p.length >= 10 ? p : [...p, b]);

  const sorted = [...shelf].sort((a, b) => {
    const sa = books.get(a.book)!.support, sb = books.get(b.book)!.support;
    return sort === "popular" ? sb - sa : sa - sb;
  });

  const found = result && result.truthRank === 1 && result.topTied === 1;
  return (
    <section className="panel" id="hide">
      <div className="panel-head">
        <div>
          <h2>Try to hide</h2>
          <p className="lede">Now you're the target. Reveal books from a real reader's shelf one at a time and see how quickly the model singles them out. Tip: popular books hide you, rare ones give you away.</p>
        </div>
      </div>
      <div className="controls">
        <button className="btn" onClick={draw}>{reader ? "Different reader" : "Pick a reader"}</button>
        {reader && (
          <div className="seg" role="group" aria-label="Sort shelf">
            <button className={sort === "popular" ? "on" : ""} onClick={() => setSort("popular")}>Most popular first</button>
            <button className={sort === "rare" ? "on" : ""} onClick={() => setSort("rare")}>Rarest first</button>
          </div>
        )}
        {picked.length > 0 && <button className="btn btn-ghost" onClick={() => setPicked([])}>Clear</button>}
      </div>

      {reader === null && <div className="empty"><p>Pick a reader to see their shelf.</p></div>}

      {reader !== null && (
        <>
          <div className={`meter ${found ? "found" : picked.length ? "hidden" : ""}`} aria-live="polite">
            {picked.length === 0 && <span>Nothing revealed yet. Reader #{fmt(reader)} is hidden among all {fmt(m.n_users)} readers.</span>}
            {result && (
              <>
                <div className="meter-big">
                  {found ? "Found." : `Hidden: ranked #${result.truthRank === Infinity ? "–" : fmt(result.truthRank!)}`}
                </div>
                <div className="meter-sub">
                  {picked.length} {picked.length === 1 ? "book" : "books"} revealed ·
                  model is {Math.round((result.truthProb ?? 0) * 100)}% sure it's Reader #{fmt(reader)} ·
                  {" "}{fmt(result.exactMatches)} {result.exactMatches === 1 ? "reader has" : "readers have"} exactly these stars
                </div>
                <div className="bar"><div className="bar-fill" style={{ width: `${Math.max(2, (result.truthProb ?? 0) * 100)}%` }} /></div>
              </>
            )}
          </div>

          {picked.length > 0 && (
            <div className="picked">
              {picked.map(b => <button key={b} className="chip" onClick={() => toggle(b)}>{shortTitle(b)} ✕</button>)}
            </div>
          )}

          <div className="shelf-grid pickable">
            {sorted.map(s => {
              const on = picked.includes(s.book);
              const b = books.get(s.book)!;
              return (
                <button key={s.book} className={`shelf-item${on ? " is-clue" : ""}`} onClick={() => toggle(s.book)}
                  aria-pressed={on} title={`${b.title} · rated by ${fmt(b.support)} readers`}>
                  <Cover id={s.book} size="sm" />
                  <Stars n={s.rating} />
                  <span className="support">{fmt(b.support)} readers</span>
                </button>
              );
            })}
          </div>
        </>
      )}
    </section>
  );
}
