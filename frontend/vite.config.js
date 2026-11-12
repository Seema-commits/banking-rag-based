import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Uncommon ports so this project doesn't clash with others on the same machine.
// Override the backend address with BACKEND_URL if you run it elsewhere.
const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8731";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5731,
    // Fail with a clear error if 5731 is taken, instead of silently using another port.
    strictPort: true,
    // Forward API calls to FastAPI during development.
    proxy: {
      "/api": BACKEND_URL,
    },
  },
});
