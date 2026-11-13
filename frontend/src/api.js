// Empty in dev (Vite proxies /api). Set VITE_API_URL for a separately hosted backend.
const BASE = import.meta.env.VITE_API_URL ?? "";

async function request(path, { adminKey, headers, ...options } = {}) {
  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers: {
      ...(adminKey ? { "X-Admin-Key": adminKey } : {}),
      ...headers,
    },
  });
  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") message = body.detail;
    } catch {
      /* non-JSON error body */
    }
    const err = new Error(message);
    err.status = res.status;
    throw err;
  }
  return res.status === 204 ? null : res.json();
}

export const api = {
  // History is kept on the server; the client only sends the session id it got back.
  chat: (question, sessionId) =>
    request("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: sessionId }),
    }),

  suggestions: () => request("/api/suggestions"),

  // Same as chat(), but calls onStep(stepName) as each agent step starts. Resolves to the answer.
  chatStream: async (question, sessionId, onStep) => {
    const res = await fetch(`${BASE}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: sessionId }),
    });
    if (!res.ok || !res.body) {
      let message = `Request failed (${res.status})`;
      try {
        const body = await res.json();
        if (typeof body.detail === "string") message = body.detail;
      } catch {
        /* non-JSON error body */
      }
      throw new Error(message);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let end;
      while ((end = buffer.indexOf("\n\n")) >= 0) {
        const line = buffer.slice(0, end).replace(/^data: /, "");
        buffer = buffer.slice(end + 2);
        const event = JSON.parse(line);
        if (event.type === "step") onStep?.(event.step);
        else if (event.type === "result") return event;
        else if (event.type === "error") throw new Error(event.detail || "Request failed");
      }
    }
    throw new Error("The connection closed before the answer arrived. Please try again.");
  },

  feedback: (messageId, rating, comment) =>
    request("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message_id: messageId, rating, comment: comment || null }),
    }),

  insights: (adminKey) => request("/api/admin/insights", { adminKey }),

  // Returns the JSONL text of unanswered/downvoted questions for the eval set.
  evalCandidates: async (adminKey) => {
    const res = await fetch(`${BASE}/api/admin/insights/eval-candidates`, {
      headers: { "X-Admin-Key": adminKey },
    });
    if (!res.ok) throw new Error(`Export failed (${res.status})`);
    return res.text();
  },

  verifyAdmin: (adminKey) => request("/api/admin/verify", { adminKey }),

  listDocuments: (adminKey) => request("/api/admin/documents", { adminKey }),

  uploadDocuments: (adminKey, files) => {
    const form = new FormData();
    for (const f of files) form.append("files", f);
    return request("/api/admin/documents", { method: "POST", body: form, adminKey });
  },

  deleteDocument: (adminKey, id) =>
    request(`/api/admin/documents/${encodeURIComponent(id)}`, {
      method: "DELETE",
      adminKey,
    }),
};
