import { useState } from "react";
import Chat from "./components/Chat.jsx";
import Admin from "./components/Admin.jsx";

// Branding comes from frontend/.env (see .env.example); set VITE_DISCLAIMER to "" to hide the notice.
const TITLE = import.meta.env.VITE_APP_TITLE || "Thistlewood Bank Assistant";
const DISCLAIMER =
  import.meta.env.VITE_DISCLAIMER ??
  "Portfolio demo. Thistlewood Bank is fictional and not affiliated with any bank or the RBI. Not financial advice. Do not enter personal or account details.";
document.title = TITLE;

export default function App() {
  const [view, setView] = useState("chat");

  return (
    <div className="app">
      <header className="topbar">
        <h1>{TITLE}</h1>
        <nav>
          <button
            className={view === "chat" ? "tab active" : "tab"}
            onClick={() => setView("chat")}
          >
            Chat
          </button>
          <button
            className={view === "admin" ? "tab active" : "tab"}
            onClick={() => setView("admin")}
          >
            Admin
          </button>
        </nav>
      </header>
      {DISCLAIMER && <p className="disclaimer">{DISCLAIMER}</p>}
      <main className="content">{view === "chat" ? <Chat /> : <Admin />}</main>
    </div>
  );
}
