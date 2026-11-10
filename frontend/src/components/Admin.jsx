import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api.js";

const KEY_STORAGE = "ragbased.adminKey";

function readStoredKey() {
  try {
    return sessionStorage.getItem(KEY_STORAGE) || "";
  } catch {
    return "";
  }
}

export default function Admin() {
  const [adminKey, setAdminKey] = useState(readStoredKey);
  const [authed, setAuthed] = useState(false);
  const [keyInput, setKeyInput] = useState("");
  const [docs, setDocs] = useState([]);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState(null); // { type: "ok" | "error", text }
  const fileRef = useRef(null);

  const logout = useCallback(() => {
    try {
      sessionStorage.removeItem(KEY_STORAGE);
    } catch {
      /* ignore */
    }
    setAdminKey("");
    setAuthed(false);
    setDocs([]);
  }, []);

  const refresh = useCallback(
    async (key = adminKey) => {
      try {
        setDocs(await api.listDocuments(key));
      } catch (err) {
        if (err.status === 401) logout();
        setStatus({ type: "error", text: err.message });
      }
    },
    [adminKey, logout]
  );

  // Re-validate a key remembered from earlier in this browser session.
  useEffect(() => {
    if (!adminKey) return;
    api
      .verifyAdmin(adminKey)
      .then(() => {
        setAuthed(true);
        refresh(adminKey);
      })
      .catch(logout);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function login(e) {
    e.preventDefault();
    setStatus(null);
    try {
      await api.verifyAdmin(keyInput);
      try {
        sessionStorage.setItem(KEY_STORAGE, keyInput);
      } catch {
        /* ignore */
      }
      setAdminKey(keyInput);
      setAuthed(true);
      setKeyInput("");
      refresh(keyInput);
    } catch (err) {
      setStatus({ type: "error", text: err.message });
    }
  }

  async function upload(e) {
    e.preventDefault();
    const files = Array.from(fileRef.current?.files || []);
    if (files.length === 0) return;
    setBusy(true);
    setStatus({ type: "ok", text: `Uploading and indexing ${files.length} file(s)…` });
    try {
      const res = await api.uploadDocuments(adminKey, files);
      const parts = [];
      if (res.uploaded.length) parts.push(`Indexed ${res.uploaded.length} file(s).`);
      for (const f of res.failed) parts.push(`${f.filename}: ${f.error}`);
      setStatus({ type: res.failed.length ? "error" : "ok", text: parts.join(" ") });
      fileRef.current.value = "";
      refresh();
    } catch (err) {
      setStatus({ type: "error", text: err.message });
    } finally {
      setBusy(false);
    }
  }

  async function remove(doc) {
    if (!window.confirm(`Delete "${doc.filename}" from the knowledge base?`)) return;
    try {
      await api.deleteDocument(adminKey, doc.id);
      setStatus({ type: "ok", text: `Deleted ${doc.filename}.` });
      refresh();
    } catch (err) {
      setStatus({ type: "error", text: err.message });
    }
  }

  if (!authed) {
    return (
      <section className="admin">
        <form className="card login" onSubmit={login}>
          <h2>Admin sign-in</h2>
          <p>Enter the admin key configured in the backend's .env file.</p>
          <input
            type="password"
            value={keyInput}
            onChange={(e) => setKeyInput(e.target.value)}
            placeholder="Admin key"
            autoFocus
          />
          <button type="submit" disabled={!keyInput}>
            Sign in
          </button>
          {status && <p className={`status ${status.type}`}>{status.text}</p>}
        </form>
      </section>
    );
  }

  return (
    <section className="admin">
      <div className="card">
        <div className="card-head">
          <h2>Upload documents</h2>
          <button className="link" onClick={logout}>
            Sign out
          </button>
        </div>
        <form className="upload" onSubmit={upload}>
          <input ref={fileRef} type="file" accept="application/pdf,.pdf" multiple />
          <button type="submit" disabled={busy}>
            {busy ? "Indexing…" : "Upload"}
          </button>
        </form>
        {status && <p className={`status ${status.type}`}>{status.text}</p>}
      </div>

      <div className="card">
        <h2>Knowledge base ({docs.length})</h2>
        {docs.length === 0 ? (
          <p className="muted">No documents uploaded yet.</p>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>File</th>
                  <th>Pages</th>
                  <th>Chunks</th>
                  <th>Uploaded</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {docs.map((d) => (
                  <tr key={d.id}>
                    <td>{d.filename}</td>
                    <td>{d.pages}</td>
                    <td>{d.chunks}</td>
                    <td>{new Date(d.uploaded_at).toLocaleString()}</td>
                    <td>
                      <button className="danger" onClick={() => remove(d)}>
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}
