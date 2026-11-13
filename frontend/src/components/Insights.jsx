import { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";

const OUTCOME_LABELS = {
  answered: "Answered",
  not_found: "Not in documents",
  unverified: "Couldn't verify",
  needs_account_access: "Needs account access",
  financial_advice: "Advice request",
  off_topic: "Off topic",
  unsafe: "Unsafe",
};

const pct = (x) => (x == null ? "-" : `${Math.round(x * 100)}%`);
const when = (iso) => new Date(iso).toLocaleString();

export default function Insights({ adminKey }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    try {
      setData(await api.insights(adminKey));
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  }, [adminKey]);

  useEffect(() => {
    load();
  }, [load]);

  async function exportCandidates() {
    try {
      const text = await api.evalCandidates(adminKey);
      const url = URL.createObjectURL(new Blob([text], { type: "application/x-ndjson" }));
      const a = Object.assign(document.createElement("a"), { href: url, download: "eval-candidates.jsonl" });
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <div className="card">
      <div className="card-head">
        <h2>Insights</h2>
        <button className="link" onClick={load}>
          Refresh
        </button>
      </div>
      {error && <p className="status error">{error}</p>}
      {!data ? (
        <p className="muted">Loading…</p>
      ) : data.total === 0 ? (
        <p className="muted">No questions yet. Insights appear after people use the chat.</p>
      ) : (
        <>
          <div className="stats">
            <div className="stat"><b>{data.total}</b><span>questions</span></div>
            <div className="stat"><b>{pct(data.answer_rate)}</b><span>answered</span></div>
            <div className="stat"><b>{data.feedback.up} / {data.feedback.down}</b><span>👍 / 👎</span></div>
            <div className="stat"><b>{data.avg_latency_ms ?? "-"} ms</b><span>avg response</span></div>
          </div>

          <h3>Outcomes</h3>
          <div className="table-wrap">
            <table>
              <tbody>
                {Object.entries(data.by_outcome).map(([outcome, n]) => (
                  <tr key={outcome}>
                    <td>{OUTCOME_LABELS[outcome] || outcome}</td>
                    <td>{n}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h3>Knowledge gaps: questions the documents couldn't answer</h3>
          {data.unanswered.length === 0 ? (
            <p className="muted">None so far.</p>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr><th>Question</th><th>Times</th><th>Last asked</th></tr>
                </thead>
                <tbody>
                  {data.unanswered.map((u) => (
                    <tr key={u.question}>
                      <td className="wrap">{u.question}</td>
                      <td>{u.count}</td>
                      <td>{when(u.last_seen)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <h3>Answers marked 👎</h3>
          {data.downvoted.length === 0 ? (
            <p className="muted">None so far.</p>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr><th>Question</th><th>Comment</th><th>When</th></tr>
                </thead>
                <tbody>
                  {data.downvoted.map((d) => (
                    <tr key={d.id}>
                      <td className="wrap">{d.question}</td>
                      <td className="wrap">{d.comment || "-"}</td>
                      <td>{when(d.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <p className="muted">
            Export unanswered and 👎 questions as eval candidates, label them, and add them to
            eval/questions.jsonl so every failure becomes a test.
          </p>
          <button onClick={exportCandidates}>Export eval candidates</button>
        </>
      )}
    </div>
  );
}
