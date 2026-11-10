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
  chat: (question, history) =>
    request("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history }),
    }),

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
