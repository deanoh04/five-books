import { useState } from "react";
import { getModel } from "./engine";

const SERIES = [
  { key: "learned", label: "Learned model", cls: "s1" },
  { key: "rarity", label: "Rarity scoring (hand-tuned)", cls: "s2" },
  { key: "exact", label: "Exact star match", cls: "s3" },
] as const;

const W = 640, H = 300, PAD = { l: 44, r: 128, t: 16, b: 40 };

function AccuracyChart() {
  const m = getModel();
  const [noise, setNoise] = useState("0.3");
  const [hover, setHover] = useState<number | null>(null);
  const ks = m.ks;
  const x = (i: number) => PAD.l + (i / (ks.length - 1)) * (W - PAD.l - PAD.r);
  const y = (v: number) => PAD.t + (1 - v / 100) * (H - PAD.t - PAD.b);
  const row = (k: number) => m.curve[`${noise}|${k}`];
  return (
    <figure className="chart">
      <div className="chart-head">
        <figcaption>Readers correctly identified (top guess), by number of books revealed</figcaption>
        <div className="seg" role="group" aria-label="Memory noise">
          <button className={noise === "0.3" ? "on" : ""} onClick={() => setNoise("0.3")}>Imperfect memory</button>
          <button className={noise === "0.0" ? "on" : ""} onClick={() => setNoise("0.0")}>Perfect memory</button>
        </div>
      </div>
      <div className="legend">
        {SERIES.map(s => <span key={s.key}><i className={`sw ${s.cls}`} />{s.label}</span>)}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Line chart of identification accuracy"
        onMouseLeave={() => setHover(null)}>
        {[0, 25, 50, 75, 100].map(v => (
          <g key={v}>
            <line className="grid" x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} />
            <text className="tick" x={PAD.l - 8} y={y(v) + 4} textAnchor="end">{v}%</text>
          </g>
        ))}
        {ks.map((k, i) => <text key={k} className="tick" x={x(i)} y={H - PAD.b + 18} textAnchor="middle">{k}</text>)}
        <text className="axis-title" x={(PAD.l + W - PAD.r) / 2} y={H - 4} textAnchor="middle">books revealed</text>
        {hover !== null && <line className="crosshair" x1={x(hover)} x2={x(hover)} y1={PAD.t} y2={H - PAD.b} />}
        {[...SERIES].reverse().map(s => {
          const pts = ks.map((k, i) => `${x(i)},${y((row(k) as any)[s.key])}`).join(" ");
          const last = (row(ks[ks.length - 1]) as any)[s.key];
          return (
            <g key={s.key} className={s.cls}>
              <polyline className="line" points={pts} />
              {ks.map((k, i) => <circle key={k} className="dot" cx={x(i)} cy={y((row(k) as any)[s.key])} r={hover === i ? 5 : 3} />)}
              <text className="direct" x={W - PAD.r + 8} y={y(last) + (s.key === "exact" ? 4 : s.key === "rarity" ? 14 : -2)}>{s.label.split(" (")[0]}</text>
            </g>
          );
        })}
        {ks.map((_, i) => (
          <rect key={i} x={x(i) - (W - PAD.l - PAD.r) / (ks.length - 1) / 2} y={PAD.t} height={H - PAD.t - PAD.b}
            width={(W - PAD.l - PAD.r) / (ks.length - 1)} fill="transparent" onMouseEnter={() => setHover(i)} />
        ))}
      </svg>
      {hover !== null && (
        <div className="tooltip" style={{ left: `${(x(hover) / W) * 100}%` }}>
          <b>{ks[hover]} {ks[hover] === 1 ? "book" : "books"}</b>
          {SERIES.map(s => <div key={s.key}><i className={`sw ${s.cls}`} />{s.label.split(" (")[0]}: <b>{(row(ks[hover]) as any)[s.key]}%</b></div>)}
        </div>
      )}
      <table className="sr-only">
        <thead><tr><th>Books</th>{SERIES.map(s => <th key={s.key}>{s.label}</th>)}</tr></thead>
        <tbody>{ks.map(k => <tr key={k}><td>{k}</td>{SERIES.map(s => <td key={s.key}>{(row(k) as any)[s.key]}%</td>)}</tr>)}</tbody>
      </table>
    </figure>
  );
}

function DefenseChart() {
  const d = getModel().defense;
  const [hover, setHover] = useState<number | null>(null);
  const rows = d.rows;
  const xs = [0.8, 1.45];
  const x = (v: number) => PAD.l + ((v - xs[0]) / (xs[1] - xs[0])) * (W - PAD.l - PAD.r);
  const y = (v: number) => PAD.t + (1 - v / 100) * (H - PAD.t - PAD.b);
  const S = [{ key: "reid_learned", label: "Learned model", cls: "s1" }, { key: "reid_rarity", label: "Rarity scoring", cls: "s2" }];
  return (
    <figure className="chart">
      <div className="chart-head">
        <figcaption>Defense: randomize a share of ratings before release. Privacy (lower is better) vs recommendation error (lower is better)</figcaption>
      </div>
      <div className="legend">{S.map(s => <span key={s.key}><i className={`sw ${s.cls}`} />{s.label} attacker</span>)}</div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Tradeoff between re-identification and recommendation error">
        {[0, 25, 50, 75, 100].map(v => (
          <g key={v}>
            <line className="grid" x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} />
            <text className="tick" x={PAD.l - 8} y={y(v) + 4} textAnchor="end">{v}%</text>
          </g>
        ))}
        {[0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4].map(v => <text key={v} className="tick" x={x(v)} y={H - PAD.b + 18} textAnchor="middle">{v.toFixed(1)}</text>)}
        <text className="axis-title" x={(PAD.l + W - PAD.r) / 2} y={H - 4} textAnchor="middle">recommendation error (RMSE, stars)</text>
        <text className="axis-title" transform={`translate(-2 ${(PAD.t + H - PAD.b) / 2}) rotate(-90)`} textAnchor="middle">readers re-identified</text>
        <line className="ref" x1={x(d.item_mean_rmse)} x2={x(d.item_mean_rmse)} y1={PAD.t} y2={H - PAD.b} />
        <text className="ref-label" x={x(d.item_mean_rmse) + 6} y={H - PAD.b - 22}>right of this line: worse than</text>
        <text className="ref-label" x={x(d.item_mean_rmse) + 6} y={H - PAD.b - 8}>just guessing each book's average</text>
        {S.map(s => (
          <g key={s.key} className={s.cls}>
            <polyline className="line" points={rows.map(r => `${x(r.rec_rmse)},${y((r as any)[s.key])}`).join(" ")} />
            {rows.map((r, i) => (
              <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
                <circle className="dot" cx={x(r.rec_rmse)} cy={y((r as any)[s.key])} r={hover === i ? 6 : 4} />
                <circle cx={x(r.rec_rmse)} cy={y((r as any)[s.key])} r={14} fill="transparent" />
                {s.key === "reid_learned" && (
                  <text className="pt-label" x={x(r.rec_rmse)} y={y((r as any)[s.key]) - 10} textAnchor="middle">{Math.round(r.q * 100)}%</text>
                )}
              </g>
            ))}
          </g>
        ))}
      </svg>
      {hover !== null && (
        <div className="tooltip" style={{ left: `${(x(rows[hover].rec_rmse) / W) * 100}%` }}>
          <b>{Math.round(rows[hover].q * 100)}% of ratings randomized</b>
          <div><i className="sw s1" />Learned attacker: <b>{rows[hover].reid_learned}%</b> re-identified</div>
          <div><i className="sw s2" />Rarity attacker: <b>{rows[hover].reid_rarity}%</b></div>
          <div>Recommender error: <b>{rows[hover].rec_rmse.toFixed(3)}</b></div>
        </div>
      )}
      <p className="chart-note">Labels on the blue line show the share of ratings randomized. Attack uses 5 books with imperfect memory.</p>
    </figure>
  );
}

export default function Results() {
  const m = getModel();
  const h = m.headline ?? {};
  const k5 = h.k5_noise30 ?? { learned: m.curve["0.3|5"].learned, exact: m.curve["0.3|5"].exact };
  const pop = h.k5_noise30_popular;
  const def = m.defense.rows;
  const q75 = def[def.length - 1];
  return (
    <section className="panel" id="results">
      <h2>Results</h2>
      <p className="lede">All numbers come from readers the learned model never trained on (3,000 for the headline figures, 600 per point on the charts). Unless noted, the attacker knows 5 books from the target's shelf with 30% of the stars misremembered by one.</p>
      <div className="tiles">
        <div className="tile"><span className="tile-num">{k5.learned}%</span><span className="tile-label">identified from 5 books, learned model</span></div>
        <div className="tile"><span className="tile-num">{k5.exact}%</span><span className="tile-label">exact star matching on the same test</span></div>
        <div className="tile"><span className="tile-num">{q75.reid_learned}%</span><span className="tile-label">still identified after randomizing 75% of the database</span></div>
      </div>
      <AccuracyChart />
      <DefenseChart />
      <div className="findings">
        <h3>What this shows</h3>
        <ul>
          <li><b>Rare books give you away.</b> Weighting each match by how few readers share that book is what makes 5 facts enough. Exact matching collapses once memory is imperfect.</li>
          <li><b>The learned model beats the hand-tuned score</b> most where it's hardest: 3 books with noisy stars, and a noised database, where it stays far ahead.</li>
          <li><b>Naive noise is a bad defense.</b> Randomizing ratings makes recommendations worse than a trivial baseline long before it stops re-identification. {q75 && `At 75% randomized, the learned attacker still finds ${q75.reid_learned}% of readers.`}</li>
          {pop && <li><b>Mainstream readers are safer.</b> If the attacker only knows your 5 most popular books, accuracy drops to {pop.learned}%{h.k8_noise30_popular ? `, and 8 of them gets it back to ${h.k8_noise30_popular.learned}%` : ""}. That's the honest limit of this attack.</li>}
        </ul>
      </div>
    </section>
  );
}
