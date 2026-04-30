# ITS RAG — React Frontend

React + TypeScript + Vite UI for the Intelligent Ticketing System RAG pipeline.
Communicates with the FastAPI backend (`api.py`) running on port 8000.

---

## Quick Start

```bash
# From the frontend/ directory
npm install
npm run dev
# Opens at http://localhost:5173
```

The backend must be running first:

```bash
# From the project root
uvicorn api:app --reload --port 8000
```

---

## Structure

```
frontend/
├── src/
│   ├── App.tsx            # Main app — form, submit handler, state
│   ├── App.css            # App-level styles
│   ├── index.css          # Global styles
│   ├── main.tsx           # React entry point
│   ├── api.ts             # API client — calls FastAPI /api/status and /api/analyze
│   ├── types.ts           # TypeScript interfaces (SystemStatus, AnalyzeResult, etc.)
│   ├── assets/            # Static assets
│   ├── components/
│   │   ├── ResultPanel.tsx  # Displays resolution blueprint, similar tickets, KB articles
│   │   ├── Sidebar.tsx      # System status (retriever ready, LLM online, ticket count)
│   │   ├── TopBar.tsx       # Navigation/header
│   │   ├── NeuralBg.tsx     # Animated canvas background
│   │   └── CursorGlow.tsx   # Cursor glow effect
│   └── hooks/             # Custom React hooks
├── package.json
├── vite.config.ts
├── tsconfig.app.json
└── tsconfig.node.json
```

---

## API Integration

All backend calls are in [src/api.ts](src/api.ts).

**`getStatus()`** — polls `GET /api/status` to show system health in the sidebar.

**`analyze(description, topKTickets, topKKb, useReranking)`** — posts to `POST /api/analyze`, returns the resolution blueprint, similar tickets, KB articles, and timings.

The base URL defaults to `http://localhost:8000`. Change it in `api.ts` if your backend runs on a different host/port.

---

## Key Components

**[ResultPanel.tsx](src/components/ResultPanel.tsx)** — renders the full resolution blueprint returned by the API, including:
- Problem restatement
- Resolution paragraphs
- Numbered troubleshooting steps
- Similar historical tickets with scores
- KB article excerpts
- Escalation conditions
- User email draft

**[Sidebar.tsx](src/components/Sidebar.tsx)** — shows live system status fetched from `/api/status`:
- Retriever ready / error
- Ticket count and KB count
- LLM online status and model name

---

## TypeScript Types

Defined in [src/types.ts](src/types.ts):

```ts
interface SystemStatus {
  retriever_ready: boolean;
  retriever_error: string | null;
  ticket_count: number | null;
  kb_count: number | null;
  llm_ready: boolean;
  llm_model: string;
}

interface AnalyzeResult {
  resolution: string;          // Markdown-formatted blueprint
  similar_tickets: SimilarTicket[];
  kb_articles: KbArticle[];
  timings: { tickets_s: number; kb_s: number; llm_s: number; total_s: number };
  counts: { tickets: number; kb: number };
}
```

---

## Build for Production

```bash
npm run build
# Output in frontend/dist/
```

Serve the `dist/` folder with any static file server, or point a reverse proxy (nginx, Caddy) to it alongside the FastAPI backend.

---

## Development Notes

- Hot module replacement (HMR) is enabled via Vite.
- ESLint is configured. Run `npm run lint` to check.
- The backend enables CORS for all origins (`*`), so no proxy config is needed during development.
