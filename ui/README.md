# Sentinel v2 dashboard

React 19 + TypeScript + Vite single-page app for the Sentinel paper-trading lab. It talks to the
FastAPI backend described in `../docs/API.md` and ships a deterministic mock of every endpoint
for offline work.

## Commands

| Command | What it does |
|---|---|
| `npm run dev` | Dev server on http://localhost:5173 with `/api` proxied to `http://127.0.0.1:8787`. If the backend is down, the UI falls back to demo data and shows a banner. |
| `npm run dev:mock` | Same, but forces the mock API (`--mode mock` loads `.env.mock`, which sets `VITE_MOCK=1`). |
| `npm run build` | Type-checks (`tsc -b`, strict) and builds to `dist/` with `base: '/'`. The backend serves `dist/`. |
| `npm run preview` | Serves the production build locally. |
| `npm run lint` | oxlint. |

From the repo root, `.claude/launch.json` defines `ui-mock` (mock dev server) and `ui` (real backend) for the Claude Code preview pane.

## Layout

```
src/
  api/        types.ts (API contract), contract.ts (ApiClient interface), client.ts (fetchers, mock switch, useSSE),
              mock.ts (seeded demo world + fake stream), hooks.ts (usePoll / useAction / useNow)
  state/      AppContext.tsx (status via SSE, connection state, live events + equity, polling fallback)
  lib/        format.ts (money / % / R helpers), time.ts (America/New_York helpers), theme.ts (dark/light store), nav.ts
  components/ ui/ (Card, Badge, Button, DataTable, Drawer, Confirm, Toast, Chips, Gauge, KpiTile, ...)
              charts/ (EquityChart, PnlCurve, BarCharts, Heatmap), layout/ (AppShell, Sidebar, TopBar)
              Markdown.tsx (safe markdown-to-React), drawers and cards used by pages
  pages/      Overview, Tournament, Trades, Factors, Research, Settings
```

## Theme

Tokens live in `src/index.css` under `:root` (light) and `.dark`, exposed to Tailwind v4 through
`@theme inline`. The theme is persisted in `localStorage` (`sentinel.theme`) and follows
`prefers-color-scheme` on first load; `index.html` applies it before first paint.
