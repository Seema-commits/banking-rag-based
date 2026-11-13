import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "../api.js";

// Raw HTML is never rendered (safer), so turn stray <br> tags from the LLM into plain separators.
const cleanAnswer = (text) => text.replace(/<br\s*\/?>/gi, " ");

// Wide tables scroll inside the bubble instead of stretching it.
const markdownComponents = {
  table: ({ node, ...props }) => (
    <div className="table-wrap">
      <table {...props} />
    </div>
  ),
};

function Feedback({ message, onRate }) {
  if (!message.messageId) return null;
  if (message.rating) {
    return <p className="feedback-done">Thanks for your feedback.</p>;
  }
  return (
    <div className="feedback" aria-label="Was this answer helpful?">
      <button type="button" className="icon" title="Helpful" onClick={() => onRate(message, "up")}>
        👍
      </button>
      <button type="button" className="icon" title="Not helpful" onClick={() => onRate(message, "down")}>
        👎
      </button>
    </div>
  );
}

export default function Chat() {
  // Each message: { role, content, sources?, notice?, outcome?, messageId?, rating?, error? }
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const sessionId = useRef(null); // conversation id; the server keeps the history
  const bottomRef = useRef(null);

  function newChat() {
    sessionId.current = null;
    setMessages([]);
  }

  async function rate(message, rating) {
    const comment = rating === "down" ? window.prompt("What was wrong? (optional)") : null;
    try {
      await api.feedback(message.messageId, rating, comment);
      setMessages((prev) => prev.map((m) => (m === message ? { ...m, rating } : m)));
    } catch {
      /* feedback is best-effort */
    }
  }

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function send(e) {
    e.preventDefault();
    const question = input.trim();
    if (!question || loading) return;

    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setInput("");
    setLoading(true);
    try {
      const res = await api.chat(question, sessionId.current);
      sessionId.current = res.session_id;
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.answer,
          sources: res.sources,
          notice: res.notice,
          outcome: res.outcome,
          messageId: res.message_id,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: err.message, error: true },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) send(e);
  }

  return (
    <section className="chat">
      <div className="messages">
        {messages.length === 0 && (
          <div className="empty">
            <h2>Ask a question about the uploaded documents</h2>
            <p>Answers are based only on documents uploaded by the administrator.</p>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}${m.error ? " error" : ""}`}>
            <div className="bubble">
              {m.role === "assistant" && !m.error ? (
                <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                  {cleanAnswer(m.content)}
                </ReactMarkdown>
              ) : (
                <p>{m.content}</p>
              )}
              {m.notice && <p className="notice">{m.notice}</p>}
              {m.role === "assistant" && !m.error && <Feedback message={m} onRate={rate} />}
              {m.sources?.length > 0 && (
                <div className="sources">
                  <span>Sources:</span>
                  {m.sources.map((s, j) => (
                    <span key={`${s.doc_id}-${s.page}`} className="source">
                      [{j + 1}] {s.filename}, p. {s.page}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="msg assistant">
            <div className="bubble typing">Thinking…</div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <form className="composer" onSubmit={send}>
        {messages.length > 0 && (
          <button type="button" className="link new-chat" onClick={newChat} disabled={loading}>
            New chat
          </button>
        )}
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask a question… (Enter to send, Shift+Enter for a new line)"
          rows={2}
          maxLength={2000}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          Send
        </button>
      </form>
    </section>
  );
}
