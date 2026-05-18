"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Fraunces, JetBrains_Mono } from "next/font/google";
import {
  deleteBenchmark, listBenchmarks, type BenchmarkRun,
} from "@/lib/benchmarks";
import { useSSE, type SseEvent } from "@/lib/sse";

const display = Fraunces({
  subsets: ["latin"],
  variable: "--font-display",
  display: "swap",
  weight: ["300", "400", "500", "700", "900"],
  style: ["normal", "italic"],
});
const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
  weight: ["300", "400", "500", "700"],
});

const PRESETS = ["quick-smoke", "chat-short", "chat-long", "code-32k", "throughput"] as const;
const STATUSES = ["completed", "running", "pending", "failed", "canceled"] as const;

const STATUS_GLYPH: Record<string, string> = {
  completed: "✓",
  running: "●",
  pending: "·",
  failed: "✕",
  canceled: "⊘",
};

function fmtMetric(value: number | null, unit: string): { num: string; unit: string; full: string } {
  if (value == null) return { num: "—", unit: "", full: "—" };
  const abs = Math.abs(value);
  let num: string;
  let suffix = unit;
  if (abs >= 1_000_000) { num = (value / 1_000_000).toFixed(2); suffix = `M ${unit}`; }
  else if (abs >= 10_000) { num = (value / 1_000).toFixed(1); suffix = `K ${unit}`; }
  else if (abs >= 1_000) { num = (value / 1_000).toFixed(2); suffix = `K ${unit}`; }
  else if (abs >= 100) { num = value.toFixed(0); }
  else if (abs >= 10) { num = value.toFixed(1); }
  else { num = value.toFixed(2); }
  return { num, unit: suffix, full: `${num} ${suffix}`.trim() };
}

function fmtWhen(iso: string): { date: string; time: string; ago: string } {
  const d = new Date(iso);
  const months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"];
  const date = `${months[d.getMonth()]} ${d.getDate().toString().padStart(2, "0")}`;
  const time = `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;
  const diffSec = Math.max(1, Math.floor((Date.now() - d.getTime()) / 1000));
  let ago: string;
  if (diffSec < 60) ago = `${diffSec}s ago`;
  else if (diffSec < 3600) ago = `${Math.floor(diffSec / 60)}m ago`;
  else if (diffSec < 86400) ago = `${Math.floor(diffSec / 3600)}h ago`;
  else ago = `${Math.floor(diffSec / 86400)}d ago`;
  return { date, time, ago };
}

export default function BenchmarksPage() {
  const [runs, setRuns] = useState<BenchmarkRun[]>([]);
  const [filter, setFilter] = useState<{ deploymentName: string; presetId: string; status: string }>({
    deploymentName: "",
    presetId: "",
    status: "",
  });
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const refresh = useCallback(() => {
    listBenchmarks().then(setRuns).catch(() => {});
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  useSSE((event: SseEvent) => {
    if (
      event.type === "benchmark:created" ||
      event.type === "benchmark:status" ||
      event.type === "benchmark:deleted"
    ) {
      refresh();
    }
  });

  const filtered = useMemo(() => runs.filter((r) => {
    const depName = r.deployment?.displayName ?? r.modelName ?? "";
    if (filter.deploymentName && !depName.toLowerCase().includes(filter.deploymentName.toLowerCase())) return false;
    if (filter.presetId && r.presetId !== filter.presetId) return false;
    if (filter.status && r.status !== filter.status) return false;
    return true;
  }), [runs, filter]);

  // Rank by mean t/s descending among the visible set. Runs without a mean
  // (still pending / failed before any results) get an em-dash rank.
  const ranked = useMemo(() => {
    const withRank = filtered
      .map((r) => ({ r, score: r.meanTps ?? -1 }))
      .sort((a, b) => b.score - a.score);
    return withRank.map(({ r }, i) => ({ run: r, rank: r.meanTps != null ? i + 1 : null }));
  }, [filtered]);

  // Track the best mean t/s across visible rows so each row's tps bar reads
  // as a proportional sparkline against the leader.
  const peakTps = useMemo(() => Math.max(0, ...filtered.map((r) => r.meanTps ?? 0)), [filtered]);

  const toggle = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const cycleStatus = () => {
    const idx = STATUSES.indexOf(filter.status as (typeof STATUSES)[number]);
    const next = idx === -1 ? STATUSES[0] : idx === STATUSES.length - 1 ? "" : STATUSES[idx + 1];
    setFilter((f) => ({ ...f, status: next }));
  };
  const cyclePreset = () => {
    const idx = PRESETS.indexOf(filter.presetId as (typeof PRESETS)[number]);
    const next = idx === -1 ? PRESETS[0] : idx === PRESETS.length - 1 ? "" : PRESETS[idx + 1];
    setFilter((f) => ({ ...f, presetId: next }));
  };

  const compareHref =
    selected.size >= 2
      ? `/benchmarks/compare?ids=${Array.from(selected).join(",")}`
      : null;

  return (
    <div className={`${display.variable} ${mono.variable} bm-root`}>
      <style>{CSS}</style>
      <div className="bm-grain" aria-hidden />

      <header className="bm-masthead">
        <div className="bm-mast-row">
          <span className="bm-meta">
            VOL.&nbsp;01 &nbsp;·&nbsp; N°{filtered.length.toString().padStart(3, "0")} OF {runs.length.toString().padStart(3, "0")} &nbsp;·&nbsp; UPDATED&nbsp;LIVE
          </span>
          <span className="bm-meta bm-meta--right">
            DGX&nbsp;SPARK &nbsp;·&nbsp; INFERENCE&nbsp;TIMING
          </span>
        </div>

        <h1 className="bm-wordmark">
          <span className="bm-w bm-w-1">B</span>
          <span className="bm-w bm-w-2">E</span>
          <span className="bm-w bm-w-3">N</span>
          <span className="bm-w bm-w-4">C</span>
          <span className="bm-w bm-w-5">H</span>
          <span className="bm-w bm-w-6">M</span>
          <span className="bm-w bm-w-7">A</span>
          <span className="bm-w bm-w-8">R</span>
          <span className="bm-w bm-w-9 bm-w-accent">K</span>
          <span className="bm-w bm-w-10">S</span>
        </h1>

        <p className="bm-subhead">
          <em>Throughput, latency &amp; coherence&mdash;measured at the inference boundary,
          ranked by mean tokens per second.</em>
        </p>
      </header>

      <hr className="bm-rule" />

      <div className="bm-filters">
        <FilterChip
          label="STATUS"
          value={filter.status}
          empty="ALL"
          onClick={cycleStatus}
          onClear={() => setFilter((f) => ({ ...f, status: "" }))}
        />
        <FilterChip
          label="PRESET"
          value={filter.presetId}
          empty="ALL"
          onClick={cyclePreset}
          onClear={() => setFilter((f) => ({ ...f, presetId: "" }))}
        />
        <FilterText
          value={filter.deploymentName}
          onChange={(v) => setFilter((f) => ({ ...f, deploymentName: v }))}
        />
        <span className="bm-filter-spacer" />
        <span className="bm-meta bm-filter-result">
          {filtered.length} / {runs.length} VISIBLE
        </span>
      </div>

      <main className="bm-board" role="list">
        {ranked.length === 0 && <EmptyState hasAny={runs.length > 0} />}
        {ranked.map(({ run, rank }) => (
          <RunRow
            key={run.id}
            run={run}
            rank={rank}
            peakTps={peakTps}
            selected={selected.has(run.id)}
            onToggle={() => toggle(run.id)}
            onDelete={async () => {
              if (confirm(`Delete this benchmark run?\n\n${run.deployment?.displayName ?? run.modelName}`)) {
                await deleteBenchmark(run.id);
                refresh();
              }
            }}
          />
        ))}
      </main>

      <div className={`bm-compare ${compareHref ? "is-armed" : ""}`} aria-hidden={!compareHref}>
        <span className="bm-compare-count">
          <span className="bm-compare-n">{selected.size.toString().padStart(2, "0")}</span>
          &nbsp;RUNS&nbsp;SELECTED
        </span>
        <button
          type="button"
          className="bm-compare-clear"
          onClick={() => setSelected(new Set())}
          disabled={!compareHref}
        >
          CLEAR
        </button>
        {compareHref ? (
          <Link href={compareHref} className="bm-compare-cta">
            <em>Compare</em>
            <span className="bm-compare-arrow">→</span>
          </Link>
        ) : (
          <span className="bm-compare-cta is-ghost">
            <em>Compare</em>
            <span className="bm-compare-arrow">→</span>
          </span>
        )}
      </div>
    </div>
  );
}

function FilterChip({
  label, value, empty, onClick, onClear,
}: {
  label: string;
  value: string;
  empty: string;
  onClick: () => void;
  onClear: () => void;
}) {
  const active = value !== "";
  return (
    <button
      type="button"
      onClick={active ? onClear : onClick}
      onContextMenu={(e) => { e.preventDefault(); onClick(); }}
      className={`bm-chip ${active ? "is-active" : ""}`}
      title={active ? "click: clear · right-click: cycle" : "click: cycle"}
    >
      <span className="bm-chip-label">{label}</span>
      <span className="bm-chip-sep">›</span>
      <span className="bm-chip-value">{value || empty}</span>
    </button>
  );
}

function FilterText({
  value, onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <label className="bm-filter-text">
      <span className="bm-chip-label">MODEL</span>
      <span className="bm-chip-sep">›</span>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="ALL"
        className="bm-filter-input"
        spellCheck={false}
        autoComplete="off"
      />
    </label>
  );
}

function RunRow({
  run, rank, peakTps, selected, onToggle, onDelete,
}: {
  run: BenchmarkRun;
  rank: number | null;
  peakTps: number;
  selected: boolean;
  onToggle: () => void;
  onDelete: () => void;
}) {
  const w = fmtWhen(run.createdAt);
  const headlineName = run.deployment?.displayName ?? run.deployment?.model.name ?? run.modelName;
  const isPhantom = !run.deployment;

  const meanTps = fmtMetric(run.meanTps, "T/S");
  const meanTtfr = fmtMetric(run.meanTtfrMs, "MS");

  const tpsPct = peakTps > 0 && run.meanTps != null
    ? Math.min(100, (run.meanTps / peakTps) * 100)
    : 0;

  return (
    <article
      className={`bm-row bm-row--${run.status} ${selected ? "is-selected" : ""} ${isPhantom ? "is-phantom" : ""}`}
      role="listitem"
    >
      <button
        type="button"
        className="bm-toggle"
        onClick={onToggle}
        aria-pressed={selected}
        title={selected ? "Deselect" : "Select for compare"}
      >
        <span className="bm-toggle-bracket">[</span>
        <span className="bm-toggle-fill">{selected ? "●" : " "}</span>
        <span className="bm-toggle-bracket">]</span>
      </button>

      <div className="bm-rank" aria-label={rank ? `Rank ${rank}` : "Unranked"}>
        {rank != null ? rank.toString().padStart(2, "0") : "—"}
      </div>

      <div className="bm-headline">
        <Link href={`/benchmarks/${run.id}`} className="bm-name-link">
          <h2 className="bm-name">{headlineName}</h2>
        </Link>
        <div className="bm-byline">
          <span className="bm-byline-preset">{run.presetId ?? "custom"}</span>
          <span className="bm-byline-sep">·</span>
          <span>{w.date}&nbsp;{w.time}</span>
          <span className="bm-byline-ago">({w.ago})</span>
          <span className="bm-byline-sep">·</span>
          <span className="bm-byline-endpoint" title={run.endpointUrl}>
            {run.endpointUrl.replace(/^https?:\/\//, "")}
          </span>
        </div>
        <div className="bm-tps-bar" aria-hidden>
          <span className="bm-tps-bar-fill" style={{ width: `${tpsPct}%` }} />
        </div>
      </div>

      <div className="bm-metric">
        <span className="bm-metric-label">MEAN&nbsp;THROUGHPUT</span>
        <span className="bm-metric-num">{meanTps.num}</span>
        <span className="bm-metric-unit">{meanTps.unit || "T/S"}</span>
      </div>

      <div className="bm-metric">
        <span className="bm-metric-label">MEAN&nbsp;TTFR</span>
        <span className="bm-metric-num">{meanTtfr.num}</span>
        <span className="bm-metric-unit">{meanTtfr.unit || "MS"}</span>
      </div>

      <div className="bm-status">
        <span className="bm-status-glyph" aria-hidden>{STATUS_GLYPH[run.status] ?? "·"}</span>
        <span className="bm-status-label">{run.status}</span>
        {run.error && (
          <span className="bm-status-error" title={run.error}>
            {run.error.length > 32 ? `${run.error.slice(0, 32)}…` : run.error}
          </span>
        )}
      </div>

      <div className="bm-actions">
        <Link href={`/benchmarks/${run.id}`} className="bm-action bm-action--view">
          VIEW <span aria-hidden>→</span>
        </Link>
        <button type="button" className="bm-action bm-action--del" onClick={onDelete}>
          DEL
        </button>
      </div>
    </article>
  );
}

function EmptyState({ hasAny }: { hasAny: boolean }) {
  return (
    <div className="bm-empty">
      <p className="bm-empty-head">
        <em>{hasAny ? "Nothing matches the filter." : "Nothing to time yet."}</em>
      </p>
      <p className="bm-empty-sub">
        {hasAny
          ? "TRY CLEARING A FILTER ABOVE."
          : (<>RUN A BENCHMARK FROM <Link href="/deployments" className="bm-empty-link">/DEPLOYMENTS</Link> →</>)}
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*                              styles                                 */
/* ------------------------------------------------------------------ */

const CSS = `
.bm-root {
  --paper: #0E0B08;
  --paper-2: #14110D;
  --ink: #F2EBDD;
  --ink-2: rgba(242, 235, 221, 0.62);
  --ink-3: rgba(242, 235, 221, 0.32);
  --ink-4: rgba(242, 235, 221, 0.14);
  --rule: rgba(242, 235, 221, 0.10);
  --molten: #FF5A1F;
  --molten-soft: #FF7A47;
  --molten-glow: rgba(255, 90, 31, 0.18);
  --crimson: #FF8A7A;
  --leaf: #B8FFAB;

  position: relative;
  min-height: 100vh;
  margin: 0;
  padding: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: var(--font-display), Georgia, serif;
  font-feature-settings: "ss01", "ss02", "kern";
  letter-spacing: -0.005em;
  font-synthesis: none;
  -webkit-font-smoothing: antialiased;
  text-rendering: geometricPrecision;
  overflow-x: hidden;
}

.bm-root *,
.bm-root *::before,
.bm-root *::after { box-sizing: border-box; }

/* Subtle grain overlay for atmosphere — pure CSS via radial-gradient noise.
   Layered above the paper but below all content. */
.bm-grain {
  pointer-events: none;
  position: fixed;
  inset: 0;
  z-index: 0;
  opacity: 0.55;
  background-image:
    radial-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
    radial-gradient(rgba(0,0,0,0.04) 1px, transparent 1px);
  background-size: 3px 3px, 7px 7px;
  background-position: 0 0, 1px 2px;
  mix-blend-mode: overlay;
}

.bm-root > * { position: relative; z-index: 1; }

/* ============== Masthead ============== */

.bm-masthead {
  padding: clamp(28px, 5vw, 56px) clamp(28px, 5vw, 64px) 28px;
  max-width: 1640px;
  margin: 0 auto;
}

.bm-mast-row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 24px;
  margin-bottom: clamp(16px, 3vw, 32px);
  animation: bm-fade-in 700ms ease 80ms backwards;
}

.bm-meta {
  font-family: var(--font-mono), ui-monospace, monospace;
  font-size: 10.5px;
  font-weight: 500;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--ink-2);
}

.bm-meta--right { text-align: right; color: var(--ink-3); }

.bm-wordmark {
  margin: 0;
  font-weight: 900;
  font-style: normal;
  font-size: clamp(72px, 16vw, 220px);
  line-height: 0.86;
  letter-spacing: -0.045em;
  display: flex;
  align-items: baseline;
  font-variation-settings: "SOFT" 25, "WONK" 1, "opsz" 144;
}

.bm-w {
  display: inline-block;
  animation: bm-rise 900ms cubic-bezier(0.16, 1, 0.3, 1) backwards;
}

.bm-w-1  { animation-delay: 40ms;  }
.bm-w-2  { animation-delay: 95ms;  }
.bm-w-3  { animation-delay: 150ms; }
.bm-w-4  { animation-delay: 205ms; }
.bm-w-5  { animation-delay: 260ms; }
.bm-w-6  { animation-delay: 315ms; }
.bm-w-7  { animation-delay: 370ms; }
.bm-w-8  { animation-delay: 425ms; }
.bm-w-9  { animation-delay: 540ms; }
.bm-w-10 { animation-delay: 600ms; }

.bm-w-accent {
  color: var(--molten);
  font-style: italic;
  font-weight: 700;
  font-variation-settings: "SOFT" 100, "WONK" 1, "opsz" 144;
  margin: 0 -0.04em 0 -0.02em;
  text-shadow: 0 0 36px var(--molten-glow);
}

.bm-subhead {
  margin: clamp(20px, 3vw, 36px) 0 0;
  max-width: 56ch;
  font-weight: 300;
  font-size: clamp(15px, 1.4vw, 19px);
  line-height: 1.45;
  color: var(--ink-2);
  font-variation-settings: "opsz" 14;
  animation: bm-fade-in 800ms ease 720ms backwards;
}
.bm-subhead em {
  font-style: italic;
  font-weight: 400;
  color: var(--ink);
}

.bm-rule {
  margin: clamp(32px, 5vw, 56px) auto 0;
  max-width: 1640px;
  width: calc(100% - clamp(56px, 10vw, 128px));
  border: 0;
  border-top: 1px solid var(--rule);
  animation: bm-fade-in 700ms ease 880ms backwards;
}

/* ============== Filters ============== */

.bm-filters {
  max-width: 1640px;
  margin: 0 auto;
  padding: 20px clamp(28px, 5vw, 64px);
  display: flex;
  flex-wrap: wrap;
  gap: 28px;
  align-items: center;
  animation: bm-fade-in 700ms ease 950ms backwards;
}

.bm-chip,
.bm-filter-text {
  display: inline-flex;
  align-items: baseline;
  gap: 10px;
  background: transparent;
  border: 0;
  padding: 6px 0;
  font-family: var(--font-mono), ui-monospace, monospace;
  font-size: 11px;
  letter-spacing: 0.16em;
  color: var(--ink-2);
  text-transform: uppercase;
  cursor: pointer;
  transition: color 160ms ease;
}

.bm-chip:hover,
.bm-filter-text:hover { color: var(--ink); }

.bm-chip-label { color: var(--ink-3); font-weight: 500; }
.bm-chip-sep   { color: var(--ink-4); font-weight: 400; }
.bm-chip-value {
  color: var(--ink);
  font-weight: 700;
  letter-spacing: 0.12em;
}
.bm-chip.is-active .bm-chip-value { color: var(--molten); }
.bm-chip.is-active .bm-chip-sep   { color: var(--molten); }

.bm-filter-input {
  background: transparent;
  border: 0;
  border-bottom: 1px solid var(--ink-4);
  color: var(--ink);
  font: inherit;
  letter-spacing: 0.12em;
  font-weight: 700;
  padding: 2px 0;
  min-width: 120px;
  outline: none;
  transition: border-color 160ms ease, color 160ms ease;
}
.bm-filter-input:focus { border-bottom-color: var(--molten); }
.bm-filter-input:focus::placeholder { color: var(--molten); }
.bm-filter-input::placeholder { color: var(--ink-3); font-weight: 700; }

.bm-filter-spacer { flex: 1; }
.bm-filter-result { color: var(--ink-3); }

/* ============== Leaderboard ============== */

.bm-board {
  max-width: 1640px;
  margin: 0 auto;
  padding: 0 clamp(28px, 5vw, 64px) 200px;
  display: flex;
  flex-direction: column;
}

.bm-row {
  position: relative;
  display: grid;
  grid-template-columns: 48px 100px minmax(280px, 1fr) 200px 200px 160px 110px;
  gap: clamp(16px, 2vw, 32px);
  align-items: center;
  padding: 26px 0 24px;
  border-top: 1px solid var(--rule);
  transition: background-color 220ms ease, padding-left 220ms ease;
  animation: bm-fade-in 600ms ease backwards;
}

.bm-row::before {
  content: "";
  position: absolute;
  left: -4px;
  top: 0;
  bottom: 0;
  width: 3px;
  background: var(--molten);
  transform: scaleY(0);
  transform-origin: center;
  transition: transform 280ms cubic-bezier(0.16, 1, 0.3, 1);
}

.bm-row:hover {
  background:
    linear-gradient(90deg, rgba(255,90,31,0.025), rgba(255,90,31,0) 60%),
    var(--paper-2);
}
.bm-row:hover .bm-rank { color: var(--molten); opacity: 0.65; }
.bm-row:hover .bm-name { font-style: italic; }
.bm-row:hover::before { transform: scaleY(0.4); }

.bm-row.is-selected { background: linear-gradient(90deg, rgba(255,90,31,0.08), rgba(255,90,31,0) 70%); }
.bm-row.is-selected::before { transform: scaleY(1); }
.bm-row.is-selected .bm-rank { color: var(--molten); opacity: 1; }

.bm-row.is-phantom .bm-name { color: var(--ink-2); text-decoration: line-through 1px var(--ink-3); }

.bm-row.bm-row--running .bm-status-glyph {
  color: var(--molten);
  animation: bm-pulse 1.4s ease-in-out infinite;
}
.bm-row.bm-row--running .bm-status-label { color: var(--molten); }
.bm-row.bm-row--completed .bm-status-glyph { color: var(--leaf); }
.bm-row.bm-row--failed   .bm-status-glyph { color: var(--crimson); }
.bm-row.bm-row--failed   .bm-status-label { color: var(--crimson); }
.bm-row.bm-row--canceled .bm-status-glyph { color: var(--ink-3); }
.bm-row.bm-row--canceled .bm-status-label { color: var(--ink-3); }
.bm-row.bm-row--pending  .bm-status-glyph { color: var(--ink-3); }
.bm-row.bm-row--pending  .bm-status-label { color: var(--ink-3); font-style: italic; }

/* checkbox */
.bm-toggle {
  appearance: none;
  background: transparent;
  border: 0;
  padding: 0;
  cursor: pointer;
  font-family: var(--font-mono), monospace;
  font-size: 18px;
  font-weight: 500;
  color: var(--ink-3);
  letter-spacing: 0;
  display: inline-flex;
  align-items: center;
  gap: 0;
  transition: color 160ms ease;
}
.bm-toggle:hover { color: var(--molten); }
.bm-toggle-bracket { color: var(--ink-3); }
.bm-toggle-fill {
  display: inline-block;
  width: 12px;
  text-align: center;
  color: var(--molten);
}
.bm-row.is-selected .bm-toggle-bracket { color: var(--molten); }

/* ghost rank numeral */
.bm-rank {
  font-family: var(--font-mono), monospace;
  font-size: clamp(56px, 5.5vw, 84px);
  font-weight: 300;
  line-height: 0.9;
  letter-spacing: -0.04em;
  color: var(--ink);
  opacity: 0.22;
  text-align: right;
  font-feature-settings: "tnum" 1, "ss01" 1;
  transition: color 220ms ease, opacity 220ms ease, font-weight 220ms ease;
}

/* headline + meta + sparkline */
.bm-headline { min-width: 0; }
.bm-name-link { text-decoration: none; color: inherit; }
.bm-name-link:hover { text-decoration: none; }

.bm-name {
  margin: 0;
  font-size: clamp(28px, 3vw, 40px);
  font-weight: 400;
  line-height: 1.05;
  letter-spacing: -0.025em;
  font-variation-settings: "opsz" 40, "SOFT" 0, "WONK" 0;
  color: var(--ink);
  word-break: break-word;
  transition: font-style 220ms ease, color 160ms ease;
}
.bm-name-link:hover .bm-name { color: var(--molten); }

.bm-byline {
  margin-top: 8px;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: baseline;
  font-family: var(--font-mono), monospace;
  font-size: 10.5px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-3);
}
.bm-byline-preset { color: var(--ink-2); font-weight: 500; }
.bm-byline-sep    { color: var(--ink-4); }
.bm-byline-ago    {
  font-family: var(--font-display), serif;
  text-transform: none;
  letter-spacing: 0;
  font-style: italic;
  font-size: 13px;
  color: var(--ink-2);
}
.bm-byline-endpoint {
  color: var(--ink-3);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 320px;
}

.bm-tps-bar {
  margin-top: 14px;
  height: 1px;
  background: var(--ink-4);
  width: 100%;
  max-width: 480px;
  position: relative;
  overflow: visible;
}
.bm-tps-bar-fill {
  position: absolute;
  inset: 0 auto 0 0;
  background: var(--molten);
  box-shadow: 0 0 0 0.5px var(--molten);
  transform-origin: left;
  transition: width 600ms cubic-bezier(0.16, 1, 0.3, 1);
}

/* metric stacks */
.bm-metric {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  text-align: right;
  font-family: var(--font-mono), monospace;
  line-height: 1;
}
.bm-metric-label {
  font-size: 9.5px;
  font-weight: 500;
  letter-spacing: 0.22em;
  color: var(--ink-3);
  margin-bottom: 10px;
}
.bm-metric-num {
  font-size: clamp(28px, 2.8vw, 40px);
  font-weight: 500;
  letter-spacing: -0.02em;
  color: var(--ink);
  font-feature-settings: "tnum" 1, "ss01" 1, "zero" 1;
}
.bm-metric-unit {
  margin-top: 6px;
  font-size: 9.5px;
  font-weight: 500;
  letter-spacing: 0.22em;
  color: var(--ink-3);
  text-transform: uppercase;
}

/* status block */
.bm-status {
  display: flex;
  flex-direction: column;
  gap: 6px;
  font-family: var(--font-mono), monospace;
  font-size: 11px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--ink-2);
  min-width: 0;
}
.bm-status-glyph {
  font-size: 18px;
  line-height: 1;
  letter-spacing: 0;
  color: var(--ink-3);
}
.bm-status-label { font-weight: 500; }
.bm-status-error {
  font-family: var(--font-display), serif;
  font-size: 12px;
  font-style: italic;
  letter-spacing: 0;
  text-transform: none;
  color: var(--crimson);
  opacity: 0.85;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* actions */
.bm-actions {
  display: flex;
  flex-direction: column;
  gap: 8px;
  align-items: flex-end;
  font-family: var(--font-mono), monospace;
  font-size: 10.5px;
  letter-spacing: 0.18em;
}
.bm-action {
  background: transparent;
  border: 0;
  padding: 0;
  color: var(--ink-2);
  text-decoration: none;
  cursor: pointer;
  font: inherit;
  letter-spacing: inherit;
  text-transform: uppercase;
  transition: color 160ms ease, letter-spacing 220ms ease;
}
.bm-action--view:hover { color: var(--molten); letter-spacing: 0.24em; }
.bm-action--del { color: var(--ink-3); }
.bm-action--del:hover { color: var(--crimson); }

/* ============== Empty state ============== */

.bm-empty {
  padding: 80px 0 120px;
  text-align: center;
  border-top: 1px solid var(--rule);
}
.bm-empty-head {
  margin: 0 0 18px;
  font-size: clamp(32px, 4vw, 56px);
  font-weight: 400;
  color: var(--ink-2);
  letter-spacing: -0.02em;
  line-height: 1.05;
}
.bm-empty-head em {
  color: var(--ink);
  font-style: italic;
  font-variation-settings: "SOFT" 80, "WONK" 1, "opsz" 80;
}
.bm-empty-sub {
  font-family: var(--font-mono), monospace;
  font-size: 11px;
  letter-spacing: 0.22em;
  color: var(--ink-3);
  text-transform: uppercase;
}
.bm-empty-link {
  color: var(--molten);
  text-decoration: none;
  border-bottom: 1px solid var(--molten-glow);
  padding-bottom: 1px;
}
.bm-empty-link:hover { border-bottom-color: var(--molten); }

/* ============== Floating compare bar ============== */

.bm-compare {
  position: fixed;
  left: 50%;
  bottom: 28px;
  transform: translate(-50%, 32px);
  opacity: 0;
  pointer-events: none;
  z-index: 50;
  display: flex;
  align-items: center;
  gap: 28px;
  padding: 16px 22px 16px 28px;
  background: var(--paper);
  border: 1px solid var(--ink-4);
  box-shadow: 0 30px 80px rgba(0,0,0,0.5), 0 0 0 1px rgba(0,0,0,0.3);
  transition: transform 320ms cubic-bezier(0.16, 1, 0.3, 1), opacity 240ms ease;
}
.bm-compare.is-armed {
  transform: translate(-50%, 0);
  opacity: 1;
  pointer-events: auto;
}

.bm-compare-count {
  font-family: var(--font-mono), monospace;
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.18em;
  color: var(--ink-2);
}
.bm-compare-n { color: var(--molten); font-size: 14px; font-weight: 700; }

.bm-compare-clear {
  background: transparent;
  border: 0;
  padding: 4px 8px;
  font-family: var(--font-mono), monospace;
  font-size: 10.5px;
  letter-spacing: 0.18em;
  color: var(--ink-3);
  cursor: pointer;
  transition: color 160ms ease;
}
.bm-compare-clear:hover { color: var(--ink); }
.bm-compare-clear:disabled { opacity: 0.3; cursor: default; }

.bm-compare-cta {
  display: inline-flex;
  align-items: baseline;
  gap: 16px;
  padding: 10px 20px 10px 24px;
  font-size: 26px;
  line-height: 1;
  font-weight: 400;
  letter-spacing: -0.02em;
  color: var(--paper);
  background: var(--molten);
  text-decoration: none;
  font-variation-settings: "SOFT" 100, "WONK" 1, "opsz" 36;
  transition: background-color 200ms ease, padding 220ms cubic-bezier(0.16, 1, 0.3, 1);
}
.bm-compare-cta em {
  font-style: italic;
  font-weight: 500;
  color: inherit;
}
.bm-compare-arrow {
  font-family: var(--font-mono), monospace;
  font-size: 24px;
  letter-spacing: 0;
}
.bm-compare-cta:hover {
  background: var(--molten-soft);
  padding-right: 28px;
}
.bm-compare-cta.is-ghost {
  background: transparent;
  color: var(--ink-4);
  border: 1px solid var(--ink-4);
  cursor: default;
}

/* ============== Animations ============== */

@keyframes bm-rise {
  from { opacity: 0; transform: translateY(40px) rotate(-1deg); }
  to   { opacity: 1; transform: translateY(0) rotate(0); }
}
@keyframes bm-fade-in {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes bm-pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%      { opacity: 0.45; transform: scale(0.85); }
}

/* ============== Responsive ============== */

@media (max-width: 1180px) {
  .bm-row {
    grid-template-columns: 36px 80px minmax(220px, 1fr) 160px 160px 130px 90px;
    gap: 18px;
  }
  .bm-byline-endpoint { max-width: 220px; }
}

@media (max-width: 900px) {
  .bm-row {
    grid-template-columns: 28px 64px 1fr 130px 110px;
    grid-template-areas:
      "tg rank head meanA meanB"
      "tg rank byline byline byline"
      "tg rank status status actions";
    gap: 14px 16px;
  }
  .bm-toggle      { grid-area: tg; align-self: start; padding-top: 10px; }
  .bm-rank        { grid-area: rank; align-self: start; padding-top: 4px; }
  .bm-headline    { grid-area: head / head / head / head; }
  .bm-metric:nth-of-type(1) { grid-area: meanA; }
  .bm-metric:nth-of-type(2) { grid-area: meanB; }
  .bm-status      { grid-area: status; flex-direction: row; align-items: baseline; gap: 12px; }
  .bm-actions     { grid-area: actions; flex-direction: row; }
}
`;
