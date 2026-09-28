import { useEffect, useState } from "react";
import { loadMeta, Model } from "./engine";
import Mystery from "./Mystery";
import Hide from "./Hide";
import Results from "./Results";

const fmt = (n: number) => n.toLocaleString("en-US");

export default function App() {
  const [model, setModel] = useState<Model | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { loadMeta().then(setModel).catch(e => setErr(String(e))); }, []);

  if (err) return <main className="wrap"><p className="error">Couldn't load the data: {err}</p></main>;
  if (!model) return <main className="wrap loading"><span className="spinner" /> Loading {""}the library…</main>;

  return (
    <>
      <header className="hero wrap">
        <p className="eyebrow">A re-identification experiment on real Goodreads data</p>
        <h1>Five books is enough.</h1>
        <p className="hero-sub">
          Name five books a person has rated, even with a few stars misremembered, and this model picks
          them out of <b>{fmt(model.n_users)}</b> real readers. It gets it right{" "}
          <b>{model.headline?.k5_noise30?.learned ?? model.curve["0.3|5"].learned}%</b> of the time.
          Don't take our word for it. Test it below.
        </p>
        <nav className="jump">
          <a href="#mystery">Mystery reader</a>
          <a href="#hide">Try to hide</a>
          <a href="#results">Results</a>
          <a href="#how">How it works</a>
        </nav>
      </header>
      <main className="wrap">
        <Mystery />
        <Hide />
        <Results />
        <section className="panel prose" id="how">
          <h2>How it works</h2>
          <p><b>The threat.</b> Someone holds an "anonymized" ratings database with names removed, and knows a few things about you: books you mentioned, roughly how you rated them. Can they find your row? This is the setup of Narayanan &amp; Shmatikov's 2008 study that re-identified users in the Netflix Prize data.</p>
          <p><b>The data.</b> {fmt(model.n_ratings)} ratings of the {fmt(model.n_books)} most popular books by {fmt(model.n_users)} Goodreads readers (goodbooks-10k). Readers are numbered, not named.</p>
          <p><b>The model.</b> For every reader who shares at least one clue book, it builds six features: matches weighted by how rare each book is (exact, off by one star, big mismatch), raw match counts, and shelf size, since big shelves match by chance more often. A conditional logit model, trained on 3,000 simulated attacks against the hardest look-alike readers, learns how to weigh them. Scores become a confidence through a softmax with a fitted temperature, so "90% confident" really means right about 90% of the time (calibration error about 1 point on held-out readers). Training and test readers never overlap.</p>
          <p><b>Everything runs in your browser.</b> The database is split into small binary shards, so a search downloads only the few hundred KB it needs and scores all {fmt(model.n_users)} readers locally in milliseconds. No server, no tracking.</p>
          <p className="muted small">Data: goodbooks-10k by Zygmunt Zając, CC BY-SA 4.0. Built by Dean Oh.</p>
        </section>
      </main>
    </>
  );
}
