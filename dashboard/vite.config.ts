import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const API_TARGET = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/ws": { target: API_TARGET, ws: true },
      "/sim": API_TARGET,
      "/orders": API_TARGET,
      "/prediction": API_TARGET,
      "/dispatch": API_TARGET,
      "/analysis": API_TARGET,
      "/experiments": API_TARGET,
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
