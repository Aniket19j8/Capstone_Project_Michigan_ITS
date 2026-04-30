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
}

function StatusRow({ ok, label }: { ok: boolean; label: string }) {
  return (
    <div
      className={`flex items-start gap-2.5 rounded-lg px-3 py-2.5 text-xs border
        ${ok
          ? "bg-green-500/10 border-green-500/20 text-green-300"
          : "bg-red-500/10 border-red-500/20 text-red-300"
        }`}
    >
      <span className={`mt-0.5 w-2 h-2 rounded-full flex-shrink-0 ${ok ? "bg-green-400" : "bg-red-400"}`} />
      <span className="leading-snug">{label}</span>
    </div>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <label className="text-xs text-gray-400">{label}</label>
        <span className="text-xs font-semibold text-accent-light bg-accent/15 px-2 py-0.5 rounded">
          {value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-1.5 rounded-full appearance-none cursor-pointer
          bg-surface-4 accent-[#7c3aed]"
      />
      <div className="flex justify-between text-[10px] text-gray-700 mt-0.5">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

export default function Sidebar({
  status,
  statusError,
  topKTickets,
  topKKb,
  useReranking,
  onTopKTickets,
  onTopKKb,
  onUseReranking,
}: Props) {
  return (
    <aside className="w-64 flex-shrink-0 border-r border-border bg-surface-1 flex flex-col overflow-y-auto">
      {/* System Status */}
      <div className="px-4 py-4 border-b border-border">
        <h3 className="text-[10px] uppercase tracking-widest text-gray-500 font-semibold mb-3">
          System Status
        </h3>
        <div className="space-y-2">
          {statusError ? (
            <StatusRow ok={false} label={`Backend unreachable: ${statusError}`} />
          ) : status ? (
            <>
              <StatusRow
                ok={status.retriever_ready}
                label={
                  status.retriever_ready
                    ? `Retriever Ready — Tickets: ${status.ticket_count ?? "?"} · KB: ${status.kb_count ?? "?"}`
                    : `Retriever Failed: ${status.retriever_error ?? "unknown error"}`
                }
              />
              <StatusRow
                ok={status.llm_ready}
                label={
                  status.llm_ready
                    ? `LLM Ready: ${status.llm_model}`
                    : `LLM Offline — run \`ollama serve\``
                }
              />
            </>
          ) : (
            <div className="flex items-center gap-2 text-xs text-gray-500 px-3 py-2">
              <span className="w-2 h-2 rounded-full bg-gray-600 animate-pulse" />
              Checking status…
            </div>
          )}
        </div>
      </div>

      {/* Divider */}
      <div className="border-b border-border" />

      {/* Settings */}
      <div className="px-4 py-4 space-y-5">
        <h3 className="text-[10px] uppercase tracking-widest text-gray-500 font-semibold">
          Settings
        </h3>

        <Slider
          label="Similar tickets"
          value={topKTickets}
          min={1}
          max={10}
          onChange={onTopKTickets}
        />
        <Slider
          label="KB articles"
          value={topKKb}
          min={1}
          max={10}
          onChange={onTopKKb}
        />

        {/* Reranking toggle */}
        <div className="flex items-center justify-between">
          <label className="text-xs text-gray-400 cursor-pointer" htmlFor="rerank-toggle">
            Use reranking
          </label>
          <button
            id="rerank-toggle"
            role="switch"
            aria-checked={useReranking}
            onClick={() => onUseReranking(!useReranking)}
            className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors
              ${useReranking ? "bg-accent" : "bg-surface-4"}`}
          >
            <span
              className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow transition-transform
                ${useReranking ? "translate-x-4" : "translate-x-1"}`}
            />
          </button>
        </div>
      </div>
    </aside>
  );
}
