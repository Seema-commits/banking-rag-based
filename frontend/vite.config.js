import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// Ports come from frontend/.env (see .env.example), not from this file.
//   FRONTEND_PORT  preferred dev-server port; if it's busy, Vite uses the next free one
//   BACKEND_URL    where the FastAPI backend runs (must match PORT in backend/.env)
export default defineConfig(({ mode }) => {
  const env = { ...loadEnv(mode, process.cwd(), ""), ...process.env };
  return {
    plugins: [react()],
    server: {
      port: Number(env.FRONTEND_PORT) || 5175,
      // Forward API calls to FastAPI during development.
      proxy: {
        "/api": env.BACKEND_URL || "http://localhost:8731",
      },
    },
  };
});
