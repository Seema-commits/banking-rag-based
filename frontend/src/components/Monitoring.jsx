import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api.js";

// Owner dashboard: health, trends, where the time goes, problems with suggested fixes.
// Data comes from GET /api/admin/monitoring (the audit log), so it survives restarts when
// permanent storage is on.

const RANGES = [
  { hours: 24, label: "24 hours" },
  { hours: 24 * 7, label: "7 days" },
  { hours: 24 * 30, label: "30 days" },
];

// Outcome series: fixed order and colours (validated for colour-vision deficiency, light + dark).
const OUTCOME_SERIES = [
  { key: "answered", label: "Answered", color: "var(--viz-answered)" },
  { key: "not_answered", label: "Not answered", color: "var(--viz-not-answered)" },
  { key: "error", label: "Errors", color: "var(--viz-error)" },
];
const LATENCY_SERIES = [
  { key: "p50_ms", label: "Typical (median)", color: "var(--viz-series-1)" },
  { key: "p95_ms", label: "Slowest 5%", color: "var(--viz-series-2)" },
];

const OUTCOME_LABELS = {
  answered: "Answered", answered_web: "Answered from web", small_talk: "Small talk",
  not_found: "Not in documents", unverified: "Couldn't verify", needs_account_access: "Needs account access",
  financial_advice: "Advice request", off_topic: "Off topic", unsafe: "Unsafe", other_bank: "Other bank",
  llm_auth_error: "Groq key rejected", llm_rate_limited: "Groq rate limit", llm_not_configured: "Groq key missing",
  error: "Error",
};
const STEP_NAMES = {
  rewrite: "Follow-up rewrite", retrieval: "Document search", grade: "Relevance check",
  query_rewrite: "Search retry", web_search: "Web search", generate: "Writing the answer", grounding: "Fact-check",
};
const SEVERITY = {
  critical: { icon: "⛔", label: "Critical" },
  warning: { icon: "⚠️", label: "Warning" },
  info: { icon: "ℹ️", label: "Info" },
};

const secs = (ms) => (ms == null ? "–" : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`);
const pct = (x) => (x == null ? "–" : `${Math.round(x * 100)}%`);
const num = (n) => (n == null ? "–" : n.toLocaleString());
const when = (iso) => new Date(iso).toLocaleString();

function bucketLabel(iso, hours) {
  const d = new Date(iso);
  return hours <= 48
    ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : d.toLocaleDateString([], { day: "numeric", month: "short" });
}

function niceMax(v) {
  if (!v) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  return Math.ceil(v / p) * p;
}

// --------------------------------------------------------------------------- chart pieces

const W = 640;
const H = 220;
const M = { top: 10, right: 12, bottom: 24, left: 44 };
const IW = W - M.left - M.right;
const IH = H - M.top - M.bottom;

function Legend({ series }) {
  return (
    <div className="viz-legend">
      {series.map((s) => (
        <span key={s.key}>
          <i style={{ background: s.color }} /> {s.label}
        </span>
      ))}
    </div>
  );
}

function YAxis({ max, format }) {
  const ticks = [0, max / 2, max];
  return ticks.map((t) => {
    const y = M.top + IH - (t / max) * IH;
    return (
      <g key={t}>
        <line x1={M.left} x2={W - M.right} y1={y} y2={y} className="viz-grid" />
        <text x={M.left - 6} y={y + 4} textAnchor="end" className="viz-tick">{format(t)}</text>
      </g>
    );
  });
}

function XLabels({ timeline, hours, x }) {
  const every = Math.max(1, Math.ceil(timeline.length / 6));
  return timeline.map((b, i) =>
    i % every === 0 ? (
      <text key={b.start} x={x(i)} y={H - 6} textAnchor="middle" className="viz-tick">
        {bucketLabel(b.start, hours)}
      </text>
    ) : null
  );
}

function roundedTop(x, y, w, h, r) {
  const rr = Math.min(r, h, w / 2);
  return `M${x},${y + h} V${y + rr} Q${x},${y} ${x + rr},${y} H${x + w - rr} Q${x + w},${y} ${x + w},${y + rr} V${y + h} Z`;
}

function Tooltip({ tip }) {
  if (!tip) return null;
  return (
    <div className="viz-tooltip" style={{ left: `${(tip.x / W) * 100}%` }}>
      <b>{tip.title}</b>
      {tip.rows.map(([label, value, color]) => (
        <div key={label}>
          <i style={{ background: color }} /> {label}: <b>{value}</b>
        </div>
      ))}
    </div>
  );
}

function OutcomeChart({ timeline, hours }) {
  const [tip, setTip] = useState(null);
  const n = timeline.length || 1;
  const slot = IW / n;
  const barW = Math.max(2, Math.min(28, slot - 4));
  const max = niceMax(Math.max(1, ...timeline.map((b) => b.answered + b.not_answered + b.error)));
  const x = (i) => M.left + slot * i + slot / 2;
  return (
    <div className="viz-wrap">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Questions over time by outcome">
        <YAxis max={max} format={(t) => Math.round(t)} />
        {timeline.map((b, i) => {
          let y0 = M.top + IH;
          const segs = OUTCOME_SERIES.filter((s) => b[s.key] > 0);
          return (
            <g key={b.start}>
              {segs.map((s, k) => {
                const h = (b[s.key] / max) * IH;
                y0 -= h;
                const top = k === segs.length - 1;
                const gap = top ? 0 : 2; // 2px surface gap between stacked segments
                return top ? (
                  <path key={s.key} d={roundedTop(x(i) - barW / 2, y0, barW, h, 4)} fill={s.color} />
                ) : (
                  <rect key={s.key} x={x(i) - barW / 2} y={y0 + gap} width={barW} height={Math.max(0, h - gap)}
                    fill={s.color} />
                );
              })}
              <rect className="viz-hit" x={M.left + slot * i} y={M.top} width={slot} height={IH}
                onMouseEnter={() => setTip({
                  x: x(i), title: bucketLabel(b.start, hours),
                  rows: OUTCOME_SERIES.map((s) => [s.label, b[s.key], s.color]),
                })}
                onMouseLeave={() => setTip(null)} />
            </g>
          );
        })}
        <XLabels timeline={timeline} hours={hours} x={x} />
      </svg>
      <Tooltip tip={tip} />
    </div>
  );
}

function LatencyChart({ timeline, hours }) {
  const [hover, setHover] = useState(null);
  const n = timeline.length || 1;
  const slot = IW / n;
  const x = (i) => M.left + slot * i + slot / 2;
  const max = niceMax(Math.max(1000, ...timeline.flatMap((b) => [b.p50_ms || 0, b.p95_ms || 0])));
  const y = (v) => M.top + IH - (v / max) * IH;
  const path = (key) => {
    let d = "";
    let pen = false;
    timeline.forEach((b, i) => {
      if (b[key] == null) {
        pen = false;
        return;
      }
      d += `${pen ? "L" : "M"}${x(i)},${y(b[key])} `;
      pen = true;
    });
    return d;
  };
  const b = hover != null ? timeline[hover] : null;
  return (
    <div className="viz-wrap">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Response time over time">
        <YAxis max={max} format={(t) => `${(t / 1000).toFixed(t >= 10000 ? 0 : 1)}s`} />
        {LATENCY_SERIES.map((s) => (
          <path key={s.key} d={path(s.key)} fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" />
        ))}
        {/* single-point buckets still show as a dot */}
        {LATENCY_SERIES.map((s) =>
          timeline.map((p, i) =>
            p[s.key] != null && (timeline[i - 1]?.[s.key] == null) && (timeline[i + 1]?.[s.key] == null) ? (
              <circle key={`${s.key}-${i}`} cx={x(i)} cy={y(p[s.key])} r="3" fill={s.color} />
            ) : null
          )
        )}
        {b && (
          <g>
            <line x1={x(hover)} x2={x(hover)} y1={M.top} y2={M.top + IH} className="viz-crosshair" />
            {LATENCY_SERIES.map((s) =>
              b[s.key] != null ? (
                <circle key={s.key} cx={x(hover)} cy={y(b[s.key])} r="4.5" fill={s.color} className="viz-ring" />
              ) : null
            )}
          </g>
        )}
        {timeline.map((p, i) => (
          <rect key={p.start} className="viz-hit" x={M.left + slot * i} y={M.top} width={slot} height={IH}
            onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} />
        ))}
        <XLabels timeline={timeline} hours={hours} x={x} />
      </svg>
      <Tooltip tip={b && {
        x: x(hover), title: bucketLabel(b.start, hours),
        rows: LATENCY_SERIES.map((s) => [s.label, b[s.key] == null ? "no data" : secs(b[s.key]), s.color]),
      }} />
    </div>
  );
}

function StepsChart({ steps }) {
  const max = Math.max(1, ...steps.map((s) => s.avg_ms));
  return (
    <div className="steps">
      {steps.map((s) => (
        <div key={s.step} className="step-row" title={`${s.label}: ${secs(s.avg_ms)} average over ${s.runs} answers`}>
          <span className="step-label">{s.label}</span>
          <span className="step-track">
            <span className="step-bar" style={{ width: `${(s.avg_ms / max) * 100}%` }} />
          </span>
          <span className="step-value">{secs(s.avg_ms)}</span>
        </div>
      ))}
    </div>
  );
}

function ChartCard({ title, subtitle, legend, table, children, empty, wide }) {
  const [asTable, setAsTable] = useState(false);
  return (
    <div className={`card chart-card${wide ? " wide" : ""}`}>
      <div className="card-head">
        <div>
          <h3>{title}</h3>
          {subtitle && <p className="muted small">{subtitle}</p>}
        </div>
        <button className="link" onClick={() => setAsTable((t) => !t)}>{asTable ? "Show chart" : "Show as table"}</button>
      </div>
      {empty ? <p className="muted">No data in this period.</p> : asTable ? table : <>{legend}{children}</>}
    </div>
  );
}

// ------------------------------------------------------------------------------- parts

function Tile({ label, value, hint }) {
  return (
    <div className="stat" title={hint}>
      <b>{value}</b>
      <span>{label}</span>
    </div>
  );
}

function UsageMeter({ name, used, limit }) {
  const ratio = limit ? used / limit : 0;
  const level = ratio >= 1 ? "critical" : ratio >= 0.8 ? "warning" : "ok";
  return (
    <div className="meter">
      <div className="meter-head">
        <span>{name}</span>
        <span>
          {level !== "ok" && `${SEVERITY[level].icon} `}
          {num(used)} / {limit ? num(limit) : "no limit"}
        </span>
      </div>
      <div className="meter-track">
        <div className={`meter-fill ${level}`} style={{ width: `${Math.min(100, ratio * 100)}%` }} />
      </div>
    </div>
  );
}

function ReviewButton({ row, onReview }) {
  const ids = row.ids || (row.id ? [row.id] : []);
  return (
    <button className={row.reviewed ? "link small" : "small"} title={row.reviewed
      ? "Put this back on the list" : "Done with this: hide it from the list (it still counts in the stats)"}
      onClick={(e) => { e.stopPropagation(); onReview(ids, !row.reviewed); }}>
      {row.reviewed ? "Undo" : "✓ Reviewed"}
    </button>
  );
}

function ProblemTable({ title, rows, columns, onOpen, emptyText, action, onReview }) {
  const cols = onReview
    ? [...columns, { label: "", nowrap: true, render: (r) => <ReviewButton row={r} onReview={onReview} /> }]
    : columns;
  return (
    <div className="card">
      <div className="card-head">
        <h3>{title} {rows.length > 0 && <span className="count">{rows.length}</span>}</h3>
        {action}
      </div>
      {rows.length === 0 ? (
        <p className="muted">{emptyText}</p>
      ) : (
        <div className="table-wrap">
          <table className="problems">
            <thead>
              <tr>{cols.map((c, i) => <th key={c.label || i}>{c.label}</th>)}</tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={r.id || r.question || i}
                  className={[onOpen && r.id && "clickable", r.reviewed && "reviewed"].filter(Boolean).join(" ")}
                  onClick={() => onOpen && r.id && onOpen(r.id)}>
                  {cols.map((c, j) => (
                    <td key={c.label || j} className={[c.wrap && "wrap", c.nowrap && "nowrap", c.fix && "fix"].filter(Boolean).join(" ")}>
                      {c.render(r)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function Detail({ adminKey, id, onClose, onChanged }) {
  const [rec, setRec] = useState(null);
  const [error, setError] = useState(null);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const reload = useCallback(
    () => api.interaction(adminKey, id).then((r) => { setRec(r); setNote(r.review_note || ""); })
      .catch((e) => setError(e.message)),
    [adminKey, id]);
  useEffect(() => { reload(); }, [reload]);

  async function mark(body) {
    setBusy(true);
    try {
      await api.mark(adminKey, body);
      await reload();
      onChanged();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  const timings = Object.entries(rec?.timings_ms || {}).filter(([k, v]) => v > 0 && k !== "llm");
  const max = Math.max(1, ...timings.map(([, v]) => v));
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" role="dialog" aria-label="Question details" onClick={(e) => e.stopPropagation()}>
        <div className="card-head">
          <h2>Question details</h2>
          <button className="link" onClick={onClose}>Close</button>
        </div>
        {error && <p className="status error">{error}</p>}
        {!rec ? !error && <p className="muted">Loading…</p> : (
          <>
            <p className="muted small">
              {when(rec.created_at)} · {OUTCOME_LABELS[rec.outcome] || rec.outcome} · {secs(rec.latency_ms)}
              {rec.cached && " · cached"}
              {rec.reviewed && " · ✓ reviewed"}
              {rec.excluded && " · excluded from stats"}
            </p>
            <div className="review-box">
              <label htmlFor="review-note" className="small">Review note (optional)</label>
              <input id="review-note" value={note} maxLength={500} disabled={busy}
                placeholder="e.g. added a minors section to the guide" onChange={(e) => setNote(e.target.value)} />
              <div className="review-actions">
                <button disabled={busy} onClick={() => mark({ ids: [rec.id], reviewed: !rec.reviewed, note })}>
                  {rec.reviewed ? "Mark not reviewed" : "✓ Mark reviewed"}
                </button>
                <button className="secondary" disabled={busy}
                  title="For your own test questions: left out of every number and chart, still kept in the audit log"
                  onClick={() => mark({ ids: [rec.id], excluded: !rec.excluded })}>
                  {rec.excluded ? "Include in stats again" : "Exclude from stats"}
                </button>
                {rec.session_id && (
                  <button className="link" disabled={busy}
                    title="Every question in this chat, e.g. a whole test conversation"
                    onClick={() => mark({ session_id: rec.session_id, excluded: !rec.excluded })}>
                    {rec.excluded ? "Include this whole conversation" : "Exclude this whole conversation"}
                  </button>
                )}
              </div>
              {rec.reviewed && rec.reviewed_at && (
                <p className="muted small">Reviewed {when(rec.reviewed_at)}{rec.review_note ? `: “${rec.review_note}”` : ""}</p>
              )}
            </div>
            <h3>Question</h3>
            <p className="quote">{rec.question}</p>
            <h3>Answer shown</h3>
            <p className="quote pre">{rec.answer || "(no answer — the request failed)"}</p>
            {rec.sources?.length > 0 && (
              <>
                <h3>Sources</h3>
                <ul className="plain">
                  {rec.sources.map((s, i) => (
                    <li key={i}>[{i + 1}] {s.url ? <a href={s.url} target="_blank" rel="noopener noreferrer">{s.filename}</a> : `${s.filename}, p. ${s.page}`}</li>
                  ))}
                </ul>
              </>
            )}
            <h3>What the agent did</h3>
            <ul className="plain">
              {rec.small_talk && <li>Recognised small talk ({rec.small_talk}); no search.</li>}
              {rec.cached && <li>Served from the answer cache.</li>}
              {rec.rewritten && <li>Rewrote the follow-up into a standalone question.</li>}
              {rec.decision && <li>Relevance check: <b>{rec.decision.replaceAll("_", " ")}</b></li>}
              {rec.query_rewrites > 0 && <li>Retried the search with reworded query ({rec.query_rewrites}×).</li>}
              {rec.web_searched && <li>Searched official websites ({rec.web_results ?? 0} results).</li>}
              {rec.grounded != null && <li>Fact-check: <b>{rec.grounded ? "passed" : "failed"}</b>{rec.regenerations > 0 && ` (rewrote the answer ${rec.regenerations}×)`}</li>}
              {rec.error_type && <li>Error: <b>{rec.error_type}</b></li>}
              {rec.pii_masked?.length > 0 && <li>Personal data masked: {rec.pii_masked.join(", ")}</li>}
            </ul>
            {timings.length > 0 && (
              <>
                <h3>Time per step</h3>
                <div className="steps">
                  {timings.sort((a, b) => b[1] - a[1]).map(([k, v]) => (
                    <div key={k} className="step-row">
                      <span className="step-label">{STEP_NAMES[k] || k.replaceAll("_", " ")}</span>
                      <span className="step-track"><span className="step-bar" style={{ width: `${(v / max) * 100}%` }} /></span>
                      <span className="step-value">{secs(v)}</span>
                    </div>
                  ))}
                </div>
              </>
            )}
            <h3>Audit</h3>
            <p className="muted small">
              Model {rec.llm_model || "–"} · checks {rec.check_model || "–"} · prompt version {rec.prompt_version || "–"} ·
              retrieval {rec.retrieval_mode || "–"} · tokens {num((rec.tokens?.prompt_tokens || 0) + (rec.tokens?.completion_tokens || 0))}
              {rec.rating && ` · feedback ${rec.rating === "up" ? "👍" : "👎"}${rec.comment ? ` “${rec.comment}”` : ""}`}
            </p>
          </>
        )}
      </div>
    </div>
  );
}

// -------------------------------------------------------------------------------- page

export default function Monitoring({ adminKey, onUploadRequested }) {
  const [hours, setHours] = useState(24);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState(null);
  const [openId, setOpenId] = useState(null);
  const [showHidden, setShowHidden] = useState(false);

  const load = useCallback(async () => {
    try {
      setData(await api.monitoring(adminKey, hours, showHidden));
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  }, [adminKey, hours, showHidden]);

  useEffect(() => {
    load();
  }, [load]);

  const review = (ids, reviewed) => act(async () => {
    await api.mark(adminKey, { ids, reviewed });
    await load();
  }, reviewed ? "Marked reviewed. It still counts in the stats." : "Put back on the list.");

  const s = data?.summary;
  const timelineTable = useMemo(() => data && (
    <div className="table-wrap">
      <table>
        <thead><tr><th>Period</th><th>Answered</th><th>Not answered</th><th>Errors</th><th>Typical</th><th>Slowest 5%</th></tr></thead>
        <tbody>
          {data.timeline.filter((b) => b.answered + b.not_answered + b.error > 0).map((b) => (
            <tr key={b.start}>
              <td>{bucketLabel(b.start, hours)}</td><td>{b.answered}</td><td>{b.not_answered}</td><td>{b.error}</td>
              <td>{secs(b.p50_ms)}</td><td>{secs(b.p95_ms)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  ), [data, hours]);

  async function act(fn, okText) {
    try {
      await fn();
      setStatus({ type: "ok", text: okText });
    } catch (e) {
      setStatus({ type: "error", text: e.message });
    }
  }

  const open = (id) => setOpenId(id);
  const qCol = { label: "Question", wrap: true, render: (r) => r.question };
  const whenCol = { label: "When", render: (r) => when(r.created_at || r.last_seen) };

  return (
    <div className="monitoring">
      <div className="card toolbar">
        <div className="segmented" role="group" aria-label="Time range">
          {RANGES.map((r) => (
            <button key={r.hours} className={hours === r.hours ? "active" : ""} onClick={() => setHours(r.hours)}>
              {r.label}
            </button>
          ))}
        </div>
        <div className="toolbar-actions">
          <label className="toggle small">
            <input type="checkbox" checked={showHidden} onChange={(e) => setShowHidden(e.target.checked)} />
            Show hidden
          </label>
          <button className="link" onClick={load}>Refresh</button>
          <button className="link" onClick={() => act(async () => {
            const res = await api.clearCache(adminKey);
            return res;
          }, "Answer cache cleared.")}>Clear answer cache</button>
          <button className="link" onClick={() => act(() => api.download(adminKey, "/api/admin/insights/eval-candidates", "eval-candidates.jsonl"), "Eval candidates downloaded.")}>
            Export eval candidates
          </button>
          <button className="link" onClick={() => act(() => api.download(adminKey, `/api/admin/audit.csv?hours=${hours}`, "audit-log.csv"), "Audit log downloaded.")}>
            Download audit log (CSV)
          </button>
        </div>
      </div>
      {status && <p className={`status ${status.type}`}>{status.text}</p>}
      {error && <p className="status error">{error}</p>}
      {!data ? !error && <p className="muted">Loading…</p> : (
        <>
          <p className={data.storage === "firestore" ? "muted small" : "status error"}>
            {data.storage === "firestore"
              ? `History: permanent (Firestore).${data.truncated ? " Showing the latest 2,000 questions in this period." : ""}`
              : "History: temporary (local file). On Cloud Run it resets when the app restarts; see docs/storage.md."}
          </p>

          {(data.hidden.reviewed > 0 || data.hidden.excluded > 0) && !showHidden && (
            <p className="muted small">
              {data.hidden.reviewed > 0 && `${data.hidden.reviewed} reviewed ${data.hidden.reviewed === 1 ? "item is" : "items are"} hidden from the lists below. `}
              {data.hidden.excluded > 0 && `${data.hidden.excluded} test ${data.hidden.excluded === 1 ? "question is" : "questions are"} excluded from every number. `}
              <button className="link small" onClick={() => setShowHidden(true)}>Show hidden</button>
            </p>
          )}

          <div className="stats">
            <Tile label="questions" value={num(s.total)} />
            <Tile label="answered" value={pct(s.answer_rate)} hint="Answered from documents, the web, or small talk" />
            <Tile label="errors" value={num(s.errors)} />
            <Tile label="typical response" value={secs(s.p50_ms)} hint="Median, excluding cached answers" />
            <Tile label="slowest 5%" value={secs(s.p95_ms)} hint="95th percentile, excluding cached answers" />
            <Tile label="cache hits" value={pct(s.cache_hit_rate)} />
            <Tile label="👍 / 👎" value={`${s.feedback.up} / ${s.feedback.down}`} />
            <Tile label={s.estimated_cost_usd != null ? `tokens (≈ $${s.estimated_cost_usd})` : "Groq tokens"}
              value={num(s.tokens.total)} hint="Set LLM_COST_PER_MILLION_*_TOKENS to estimate cost" />
            <Tile label="web searches" value={num(s.web_searches)} />
            {s.via_mcp > 0 && <Tile label="via MCP" value={num(s.via_mcp)} hint="Asked by AI agents through /mcp" />}
          </div>

          <div className="card">
            <h3>Today's usage against limits</h3>
            <UsageMeter name="Questions" {...data.today.questions} />
            <UsageMeter name="Groq tokens" {...data.today.llm_tokens} />
            <UsageMeter name="Web searches" {...data.today.web_searches} />
          </div>

          <div className="card">
            <h3>Suggested actions</h3>
            {data.recommendations.length === 0 ? (
              <p className="muted">✅ Nothing needs attention in this period.</p>
            ) : (
              <ul className="recs">
                {data.recommendations.map((r) => (
                  <li key={r.title} className={`rec ${r.severity}`}>
                    <span className="rec-badge">{SEVERITY[r.severity].icon} {SEVERITY[r.severity].label}</span>
                    <div>
                      <b>{r.title}</b>
                      <p>{r.detail}</p>
                      {r.action === "upload" && <button onClick={onUploadRequested}>Upload a document</button>}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="chart-grid">
            <ChartCard title="Questions over time" subtitle="By outcome" empty={!s.total}
              legend={<Legend series={OUTCOME_SERIES} />} table={timelineTable}>
              <OutcomeChart timeline={data.timeline} hours={hours} />
            </ChartCard>
            <ChartCard title="Response time" subtitle="Excluding cached answers" empty={s.p50_ms == null}
              legend={<Legend series={LATENCY_SERIES} />} table={timelineTable}>
              <LatencyChart timeline={data.timeline} hours={hours} />
            </ChartCard>
            <ChartCard wide title="Where the time goes" subtitle="Average time per step, slowest first" empty={!data.steps.length}
              table={(
                <div className="table-wrap"><table>
                  <thead><tr><th>Step</th><th>Average</th><th>Answers</th></tr></thead>
                  <tbody>{data.steps.map((st) => <tr key={st.step}><td>{st.label}</td><td>{secs(st.avg_ms)}</td><td>{st.runs}</td></tr>)}</tbody>
                </table></div>
              )}>
              <StepsChart steps={data.steps} />
            </ChartCard>
          </div>

          <ProblemTable title="Errors" rows={data.errors} emptyText="No errors in this period."
            onOpen={null}
            columns={[
              { label: "Cause", nowrap: true, render: (e) => OUTCOME_LABELS[e.outcome] || e.outcome },
              { label: "Times", render: (e) => e.count },
              { label: "What to do", wrap: true, fix: true, render: (e) => e.fix },
              { label: "Recent", nowrap: true, render: (e) => e.examples.slice(0, 3).map((x, i) => (
                <button key={x.id} className="link" title={when(x.created_at)} onClick={() => open(x.id)}>
                  View {i + 1}
                </button>
              )) },
            ]} />
          <ProblemTable title={`Slow answers (over ${secs(data.slow_answer_ms)})`} rows={data.slow} onOpen={open} onReview={review}
            emptyText="No slow answers in this period."
            columns={[qCol, { label: "Time", render: (r) => secs(r.latency_ms) },
              { label: "Slowest step", render: (r) => r.slowest_step ? `${r.slowest_step} (${secs(r.slowest_step_ms)})` : "–" }, whenCol]} />
          <ProblemTable title="Knowledge gaps: questions the documents couldn't answer" rows={data.gaps} onReview={review}
            emptyText="None in this period."
            action={data.gaps.length > 0 && <button onClick={onUploadRequested}>Upload a document</button>}
            columns={[qCol, { label: "Times asked", render: (g) => g.count }, whenCol]} />
          <ProblemTable title="Answers marked 👎" rows={data.downvoted} onOpen={open} onReview={review} emptyText="None in this period."
            columns={[qCol, { label: "Comment", wrap: true, render: (r) => r.comment || "–" }, whenCol]} />
          <ProblemTable title="Answers that failed the fact-check" rows={data.unverified} onOpen={open} onReview={review}
            emptyText="None in this period." columns={[qCol, whenCol]} />
          {showHidden && (
            <ProblemTable title="Excluded from stats (test questions)" rows={data.excluded} onOpen={open}
              emptyText="Nothing excluded in this period."
              columns={[qCol, { label: "Outcome", render: (r) => OUTCOME_LABELS[r.outcome] || r.outcome }, whenCol,
                { label: "", nowrap: true, render: (r) => (
                  <button className="link small" onClick={(e) => {
                    e.stopPropagation();
                    act(async () => { await api.mark(adminKey, { ids: [r.id], excluded: false }); await load(); },
                      "Included in the stats again.");
                  }}>Include again</button>
                ) }]} />
          )}
          <p className="muted small">
            Click a row (or an error's example) to see the full record: answer, sources, the agent's steps and timings.
            ✓ Reviewed takes an item off its list once you've dealt with it; it still counts in the stats. In a
            record you can also exclude your own test questions from the stats. Nothing is deleted; turn on Show hidden to undo.
          </p>
        </>
      )}
      {openId && <Detail adminKey={adminKey} id={openId} onClose={() => setOpenId(null)} onChanged={load} />}
    </div>
  );
}
