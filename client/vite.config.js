import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: '/videogamerecs/',
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    // Ensure files are correctly hashed for caching
    assetsInlineLimit: 4096,
    // Minify the output for production (with fallback)
    minify: 'esbuild',
    // Generate source maps for debugging
    sourcemap: false,
  },
  server: {
    // Development server configuration
    port: 3000,
    strictPort: true,
    proxy: {
      // Proxy API requests to your backend during development
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      },
      // Proxy WebSocket connections
      '/ws': {
        target: 'ws://localhost:5000',
        ws: true
      }
    }
  }
})
