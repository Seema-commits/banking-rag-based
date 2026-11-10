import { useState } from "react";
import Chat from "./components/Chat.jsx";
import Admin from "./components/Admin.jsx";

export default function App() {
  const [view, setView] = useState("chat");

  return (
    <div className="app">
      <header className="topbar">
        <h1>Document Assistant</h1>
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
      <main className="content">{view === "chat" ? <Chat /> : <Admin />}</main>
    </div>
  );
}
