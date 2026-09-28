import { useEffect, useState } from "react";
import { Clue, Result, getModel, getShelf, identify } from "./engine";
import { Cover, Stars, shortTitle } from "./Cover";

type Shelf = { book: number; rating: number }[];
const fmt = (n: number) => n.toLocaleString("en-US");

function sample<T>(arr: T[], k: number) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a.slice(0, k);
}

// Same memory-noise model as training: each star is off by one with probability p.
function makeClues(shelf: Shelf, k: number, noisy: boolean): Clue[] {
  return sample(shelf, k).map(({ book, rating }) => {
    if (!noisy || Math.random() >= 0.3) return { book, rating, trueRating: rating };
    let r = rating + (Math.random() < 0.5 ? -1 : 1);
    if (r < 1 || r > 5) r = rating + (rating === 1 ? 1 : -1);
    return { book, rating: r, trueRating: rating };
  });
}

export default function Mystery() {
  const m = getModel();
  const [k, setK] = useState(5);
  const [noisy, setNoisy] = useState(true);
  const [reader, setReader] = useState<number | null>(null);
  const [shelf, setShelf] = useState<Shelf>([]);
  const [clues, setClues] = useState<Clue[]>([]);
  const [phase, setPhase] = useState<"idle" | "clues" | "searching" | "revealed">("idle");
  const [result, setResult] = useState<Result | null>(null);
  const [tally, setTally] = useState({ tries: 0, hits: 0 });
  const [counter, setCounter] = useState(0);

  async function draw() {
    const u = 1 + Math.floor(Math.random() * m.n_users);
    const s = await getShelf(u);
    setReader(u); setShelf(s); setClues(makeClues(s, k, noisy)); setResult(null); setPhase("clues");
  }

  function reshuffle(nk = k, nn = noisy) {
    if (!shelf.length) return;
    setClues(makeClues(shelf, nk, nn)); setResult(null); setPhase("clues");
  }

  async function run() {
    if (reader === null) return;
    setPhase("searching");
    const t0 = performance.now();
    const res = await identify(clues, reader);
    const wait = Math.max(0, 900 - (performance.now() - t0));
    await new Promise(r => setTimeout(r, wait));
    setResult(res);
    setPhase("revealed");
    setTally(t => ({ tries: t.tries + 1, hits: t.hits + (res.top[0]?.user === reader ? 1 : 0) }));
  }

  useEffect(() => {
    if (phase !== "searching") return;
    const start = performance.now();
    const id = setInterval(() => {
      const f = Math.min(1, (performance.now() - start) / 850);
      setCounter(Math.round(f * m.n_users));
    }, 30);
    return () => clearInterval(id);
  }, [phase, m.n_users]);

  const guess = result?.top[0];
  const correct = guess && guess.user === reader;

  return (
    <section className="panel" id="mystery">
      <div className="panel-head">
        <div>
          <h2>Mystery reader</h2>
          <p className="lede">We pick a real Goodreads reader at random and show the model only a few of their ratings. It searches all {fmt(m.n_users)} readers and names one. Then we reveal who it really was.</p>
        </div>
        <div className="tally" aria-live="polite">
          <span className="tally-num">{tally.hits}<span className="tally-den">/{tally.tries}</span></span>
          <span className="tally-label">correct this session</span>
        </div>
      </div>

      <div className="controls">
        <label className="control">
          <span>Books revealed: <b>{k}</b></span>
          <input type="range" min={1} max={10} value={k}
            onChange={e => { const v = +e.target.value; setK(v); reshuffle(v, noisy); }} />
        </label>
        <label className="toggle">
          <input type="checkbox" checked={noisy} onChange={e => { setNoisy(e.target.checked); reshuffle(k, e.target.checked); }} />
          <span>Imperfect memory <small>(30% of stars off by one)</small></span>
        </label>
        <button className="btn" onClick={draw}>{reader ? "New mystery reader" : "Draw a mystery reader"}</button>
      </div>

      {phase === "idle" && (
        <div className="empty">
          <p>Draw a reader to begin. Try it 20 times and watch the scoreboard.</p>
        </div>
      )}

      {phase !== "idle" && (
        <>
          <div className="clue-row-head">
            <span className="mono">Reader #?????</span> <span className="muted">· the model sees only this</span>
          </div>
          <div className="clues">
            {clues.map(c => (
              <div className="clue" key={c.book}>
                <Cover id={c.book} />
                <div className="clue-meta">
                  <span className="clue-title">{shortTitle(c.book)}</span>
                  <Stars n={c.rating} altered={phase === "revealed" && c.rating !== c.trueRating} />
                  {phase === "revealed" && c.rating !== c.trueRating && (
                    <span className="note">really {c.trueRating}★</span>
                  )}
                </div>
              </div>
            ))}
          </div>

          <div className="action-row">
            {phase === "clues" && <button className="btn btn-primary" onClick={run}>Identify this reader</button>}
            {phase === "searching" && (
              <div className="searching" role="status">
                <span className="spinner" /> Comparing against <span className="mono">{fmt(counter)}</span> readers…
              </div>
            )}
          </div>

          {phase === "revealed" && result && guess && (
            <div className={`verdict ${correct ? "ok" : "miss"}`}>
              <div className="verdict-main">
                <div>
                  <div className="kicker">Model's guess</div>
                  <div className="big mono">Reader #{fmt(guess.user)}</div>
                  <div className="muted">{Math.round(guess.prob * 100)}% confident{result.topTied > 1 ? ` · tied with ${result.topTied - 1} others` : ""}</div>
                </div>
                <div className="verdict-arrow" aria-hidden>→</div>
                <div>
                  <div className="kicker">Actually</div>
                  <div className="big mono">Reader #{fmt(reader!)}</div>
                  <div className={`badge ${correct ? "badge-ok" : "badge-miss"}`}>
                    {correct ? "✓ Found them" : `✗ Missed · true reader ranked #${result.truthRank === Infinity ? "–" : fmt(result.truthRank!)}`}
                  </div>
                </div>
              </div>
              <div className="verdict-foot">
                <span><b>{fmt(result.candidates)}</b> readers share at least one of these books.</span>
                <span>Exact star matching finds <b>{fmt(result.exactMatches)}</b> {result.exactMatches === 1 ? "reader" : "readers"}{result.exactMatches === 0 ? " (memory noise breaks it)" : result.exactMatches > 1 ? " (can't tell them apart)" : ""}.</span>
              </div>
            </div>
          )}

          {phase === "revealed" && (
            <details className="shelf" open>
              <summary>Reader #{fmt(reader!)}'s full shelf · {shelf.length} books (clues highlighted)</summary>
              <div className="shelf-grid">
                {[...shelf].sort((a, b) => b.rating - a.rating).map(s => (
                  <div key={s.book} className={`shelf-item${clues.some(c => c.book === s.book) ? " is-clue" : ""}`}>
                    <Cover id={s.book} size="sm" />
                    <Stars n={s.rating} />
                  </div>
                ))}
              </div>
            </details>
          )}
        </>
      )}
    </section>
  );
}
