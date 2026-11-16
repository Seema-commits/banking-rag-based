import { useEffect, useState } from "react";
import Chat from "./components/Chat.jsx";
import Admin from "./components/Admin.jsx";

// Branding comes from frontend/.env (see .env.example); set VITE_DISCLAIMER to "" to hide the notice.
const TITLE = import.meta.env.VITE_APP_TITLE || "Thistlewood Bank Assistant";
const DISCLAIMER =
  import.meta.env.VITE_DISCLAIMER ??
  "Portfolio demo. Thistlewood Bank is fictional and not affiliated with any bank or the RBI. Not financial advice. Do not enter personal or account details.";
document.title = TITLE;

// The admin page lives at <site>/#admin. Visitors to the plain link only see the chat; the admin
// page itself still needs the admin key, so hiding the button is about a tidy public page, not
// security.
const viewFromHash = () => (window.location.hash === "#admin" ? "admin" : "chat");

export default function App() {
  const [view, setView] = useState(viewFromHash);

  useEffect(() => {
    const onHash = () => setView(viewFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  function show(next) {
    // A history entry either way, so the browser's Back button works; the chat keeps a clean URL.
    if (next === "admin") window.location.hash = "admin";
    else window.history.pushState(null, "", window.location.pathname + window.location.search);
    setView(next);
  }

  return (
    <div className="app">
      <header className="topbar">
        <h1>{TITLE}</h1>
        {view === "admin" && (
          <nav>
            <button className="tab" onClick={() => show("chat")}>
              Chat
            </button>
            <button className="tab active" aria-current="page">
              Admin
            </button>
          </nav>
        )}
      </header>
      {DISCLAIMER && <p className="disclaimer">{DISCLAIMER}</p>}
      {/* The chat stays mounted (only hidden), so the conversation survives a visit to Admin. */}
      <main className="content">
        <div className="view" hidden={view !== "chat"}>
          <Chat />
        </div>
        {view === "admin" && <Admin />}
      </main>
    </div>
  );
}
