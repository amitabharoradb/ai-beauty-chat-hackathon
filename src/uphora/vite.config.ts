import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { TanStackRouterVite } from "@tanstack/router-plugin/vite";
import path from "path";

export default defineConfig({
  root: "src/uphora/ui",
  plugins: [
    TanStackRouterVite({ routesDirectory: "routes", generatedRouteTree: "types/routeTree.gen.ts" }),
    react(),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src/uphora/ui"),
    },
  },
  build: {
    outDir: path.resolve(__dirname, "src/uphora/__dist__"),
    emptyOutDir: true,
  },
});
