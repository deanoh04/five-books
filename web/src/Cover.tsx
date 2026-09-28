import { useState } from "react";
import { books } from "./engine";

// Deterministic "cloth binding" colour per book so covers without images still look like books.
const SPINES = ["#7a2e2e", "#2f4a6b", "#3d5c3a", "#6b4f2a", "#4a3a6b", "#2e5f5f", "#7a4a2e", "#555049"];

export function Cover({ id, size = "md" }: { id: number; size?: "sm" | "md" | "lg" }) {
  const b = books.get(id);
  const [failed, setFailed] = useState(false);
  if (!b) return null;
  const bg = SPINES[id % SPINES.length];
  const title = b.title.replace(/\s*\(.*?\)\s*$/, "");
  return (
    <div className={`cover cover-${size}`} style={{ background: bg }} title={`${b.title} by ${b.author}`}>
      <div className="cover-fallback">
        <span className="cover-title">{title}</span>
        <span className="cover-author">{b.author}</span>
      </div>
      {b.image && !failed && (
        <img src={b.image} alt="" loading="lazy" referrerPolicy="no-referrer" onError={() => setFailed(true)} />
      )}
    </div>
  );
}

export function Stars({ n, altered }: { n: number; altered?: boolean }) {
  return (
    <span className={`stars${altered ? " altered" : ""}`} aria-label={`${n} of 5 stars`}>
      {"★".repeat(n)}<span className="stars-off">{"★".repeat(5 - n)}</span>
    </span>
  );
}

export function shortTitle(id: number) {
  const b = books.get(id);
  return b ? b.title.replace(/\s*\(.*?\)\s*$/, "") : `Book ${id}`;
}
