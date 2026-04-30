import type { SystemStatus } from "../types";

interface Props {
  status: SystemStatus | null;
  statusError: string;
  topKTickets: number;
  topKKb: number;
  useReranking: boolean;
  onTopKTickets: (v: number) => void;
  onTopKKb: (v: number) => void;
  onUseReranking: (v: boolean) => void;
  onReset: () => void;
}

function StatusPill({ ok, children }: { ok: boolean; children: React.ReactNode }) {
  return (
    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-sm font-medium
      ${ok
        ? "bg-green-500/10 border-green-500/25 text-green-400"
        : "bg-red-500/10 border-red-500/25 text-red-400"
      }`}
    >
      <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0
        ${ok ? "bg-green-400 shadow-[0_0_6px_rgba(74,222,128,0.8)]"
             : "bg-red-400 shadow-[0_0_6px_rgba(248,113,113,0.8)]"}`}
      />
      {children}
    </div>
  );
}

function MiniSlider({ label, value, min, max, onChange }: {
  label: string; value: number; min: number; max: number; onChange: (v: number) => void;
}) {
  return (
    <div className="flex items-center gap-2.5 min-w-[160px]">
      <span className="text-sm text-gray-400 w-28 flex-shrink-0">{label}</span>
      <input
        type="range" min={min} max={max} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="flex-1 h-1 rounded-full appearance-none cursor-pointer accent-[#7c3aed]"
        style={{ accentColor: "#7c3aed" }}
      />
      <span className="text-sm font-bold text-accent-light bg-accent/20 border border-accent/30
        w-7 h-7 rounded-md flex items-center justify-center flex-shrink-0">
        {value}
      </span>
    </div>
  );
}

// ── Decorative AI brain SVG ─────────────────────────────────────────────────
function AIBrainIcon() {
  return (
    <svg width="42" height="42" viewBox="0 0 42 42" fill="none" className="opacity-90">
      <circle cx="21" cy="21" r="20" stroke="rgba(124,58,237,0.4)" strokeWidth="0.8" />
      <circle cx="21" cy="21" r="14" stroke="rgba(124,58,237,0.25)" strokeWidth="0.6" />
      {/* brain-like nodes */}
      {[
        [21,8],[33,16],[33,26],[21,34],[9,26],[9,16],
        [21,16],[28,12],[34,21],[28,30],[14,30],[8,21],[14,12]
      ].map(([cx, cy], i) => (
        <circle key={i} cx={cx} cy={cy} r="2" fill="rgba(167,139,250,0.8)"
          style={{ filter: "drop-shadow(0 0 3px rgba(124,58,237,0.9))" }} />
      ))}
      {/* connections */}
      {[
        "M21 8 L33 16","M33 16 L33 26","M33 26 L21 34",
        "M21 34 L9 26","M9 26 L9 16","M9 16 L21 8",
        "M21 16 L28 12","M21 16 L34 21","M21 16 L28 30",
        "M21 16 L14 30","M21 16 L8 21","M21 16 L14 12",
        "M21 8 L21 16","M33 16 L21 16","M33 26 L21 16",
      ].map((d, i) => (
        <path key={i} d={d} stroke="rgba(124,58,237,0.35)" strokeWidth="0.7" />
      ))}
    </svg>
  );
}

// ── RAG pipeline diagram ────────────────────────────────────────────────────
function PipelineDiagram() {
  const steps = [
    { icon: "📝", label: "Input" },
    { icon: "🔍", label: "Retrieve" },
    { icon: "🧠", label: "Rerank" },
    { icon: "✨", label: "Generate" },
  ];
  return (
    <div className="flex items-center gap-1">
      {steps.map((s, i) => (
        <div key={i} className="flex items-center gap-1">
          <div className="flex flex-col items-center gap-0.5">
            <div className="w-8 h-8 rounded-lg bg-surface-3 border border-border flex items-center justify-center text-sm">
              {s.icon}
            </div>
            <span className="text-[11px] text-gray-500">{s.label}</span>
          </div>
          {i < steps.length - 1 && (
            <div className="w-4 flex items-center justify-center mb-3.5">
              <div className="w-full h-px bg-gradient-to-r from-accent/30 to-accent/60" />
              <div className="w-1 h-1 rounded-full bg-accent/60 -ml-0.5" />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export default function TopBar({
  status, statusError,
  topKTickets, topKKb, useReranking,
  onTopKTickets, onTopKKb, onUseReranking, onReset,
}: Props) {
  return (
    <header className="relative flex-shrink-0 z-20 border-b border-border bg-surface-1/95 backdrop-blur-sm">
      {/* top gradient line */}
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-accent to-transparent" />

      {/* ── Row 1: Brand + Status + Actions ── */}
      <div className="flex items-center gap-5 px-6 py-4 border-b border-border/60">

        {/* Brand */}
        <div className="flex items-center gap-4 flex-shrink-0">
          <div className="relative">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-accent via-purple-600 to-purple-900
              flex items-center justify-center shadow-xl shadow-accent/30">
              <AIBrainIcon />
            </div>
            <span className="absolute -top-1 -right-1 w-3 h-3 rounded-full bg-green-400 border-2 border-surface-1
              shadow-[0_0_8px_rgba(74,222,128,0.8)]" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-gray-100 leading-tight tracking-tight">
              Intelligent Ticketing System
            </h1>
            <p className="text-sm text-gray-500 flex items-center gap-1.5 mt-0.5">
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
              AI-powered ticket resolution · RAG Pipeline
            </p>
          </div>
        </div>

        {/* Pipeline diagram */}
        <div className="hidden xl:flex items-center pl-4 border-l border-border">
          <PipelineDiagram />
        </div>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Status pills */}
        <div className="flex items-center gap-2 flex-wrap">
          {statusError ? (
            <StatusPill ok={false}>Backend unreachable</StatusPill>
          ) : status ? (
            <>
              <StatusPill ok={status.retriever_ready}>
                {status.retriever_ready
                  ? <>Retriever · <span className="text-white">{status.ticket_count}</span> tickets · <span className="text-white">{status.kb_count}</span> KB</>
                  : "Retriever Error"
                }
              </StatusPill>
              <StatusPill ok={status.llm_ready}>
                {status.llm_ready
                  ? <><span className="font-mono text-white">{status.llm_model}</span> · Ready</>
                  : "LLM Offline"
                }
              </StatusPill>
            </>
          ) : (
            <div className="flex items-center gap-2 text-sm text-gray-500 px-3 py-1.5 rounded-full border border-border">
              <span className="w-1.5 h-1.5 rounded-full bg-gray-600 animate-pulse" />
              Connecting…
            </div>
          )}
        </div>

        {/* Reset */}
        <button
          onClick={onReset}
          className="flex items-center gap-1.5 px-3.5 py-2 text-sm font-medium text-gray-400
            border border-border rounded-lg hover:bg-surface-3 hover:text-gray-200
            hover:border-surface-4 transition-all duration-150 flex-shrink-0"
        >
          <svg width="11" height="11" viewBox="0 0 11 11" fill="none">
            <path d="M9.5 2A5 5 0 1 0 10 5.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
            <path d="M7.5 2H9.5V4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Reset
        </button>
      </div>

      {/* ── Row 2: Settings bar ── */}
      <div className="flex items-center gap-6 px-6 py-3 bg-surface-0/60">
        <span className="text-xs uppercase tracking-widest text-gray-500 font-semibold flex-shrink-0">
          Settings
        </span>

        <div className="flex items-center gap-6 flex-wrap flex-1">
          <MiniSlider
            label="Similar tickets"
            value={topKTickets} min={1} max={10}
            onChange={onTopKTickets}
          />
          <MiniSlider
            label="KB articles"
            value={topKKb} min={1} max={10}
            onChange={onTopKKb}
          />

          {/* Reranking toggle */}
          <div className="flex items-center gap-2.5">
            <span className="text-sm text-gray-400">Use reranking</span>
            <button
              role="switch"
              aria-checked={useReranking}
              onClick={() => onUseReranking(!useReranking)}
              className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors duration-200
                ${useReranking ? "bg-accent shadow-[0_0_8px_rgba(124,58,237,0.5)]" : "bg-surface-4"}`}
            >
              <span className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow transition-transform duration-200
                ${useReranking ? "translate-x-4" : "translate-x-1"}`} />
            </button>
          </div>
        </div>

        {/* LLM model badge */}
        {status?.llm_model && (
          <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-lg
            bg-accent/10 border border-accent/20 text-sm text-accent-light flex-shrink-0">
            <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
              <circle cx="5" cy="5" r="4" stroke="currentColor" strokeWidth="0.8" />
              <circle cx="5" cy="5" r="1.5" fill="currentColor" />
              {[0,72,144,216,288].map((a, i) => (
                <circle key={i}
                  cx={5 + 3 * Math.cos(a * Math.PI / 180)}
                  cy={5 + 3 * Math.sin(a * Math.PI / 180)}
                  r="0.8" fill="currentColor" />
              ))}
            </svg>
            <span className="font-mono font-semibold">{status.llm_model}</span>
          </div>
        )}
      </div>
    </header>
  );
}
