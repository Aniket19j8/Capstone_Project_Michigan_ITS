import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Clock,
  FileText,
  Ticket,
  Sparkles,
  BookOpen,
} from "lucide-react";
import type { AnalyzeResult } from "../types";

/* ── Markdown renderer ─────────────────────────────────────────────────────── */
function MarkdownBody({ content }: { content: string }) {
  return (
    <ReactMarkdown
      components={{
        h1: ({ children }) => (
          <h1 className="text-xl font-bold text-gray-100 mt-6 mb-3 pb-2 border-b border-border">
            {children}
          </h1>
        ),
        h2: ({ children }) => (
          <h2 className="text-lg font-semibold text-gray-100 mt-6 mb-2.5 pb-1.5 border-b border-surface-4">
            {children}
          </h2>
        ),
        h3: ({ children }) => (
          <h3 className="text-base font-semibold text-accent-light mt-5 mb-2 flex items-center gap-2">
            <span className="w-1 h-5 rounded-full bg-accent inline-block flex-shrink-0" />
            {children}
          </h3>
        ),
        p: ({ children }) => (
          <p className="text-[15px] text-gray-300 leading-relaxed mb-3">{children}</p>
        ),
        strong: ({ children }) => (
          <strong className="font-semibold text-gray-100">{children}</strong>
        ),
        em: ({ children }) => (
          <em className="italic text-gray-400">{children}</em>
        ),
        ul: ({ children }) => (
          <ul className="space-y-1.5 mb-3 pl-1">{children}</ul>
        ),
        ol: ({ children }) => (
          <ol className="space-y-1.5 mb-3 pl-4 list-decimal list-inside">{children}</ol>
        ),
        li: ({ children }) => (
          <li className="text-[15px] text-gray-300 leading-relaxed flex gap-2.5 items-start">
            <span className="mt-2 w-1.5 h-1.5 rounded-full bg-accent flex-shrink-0" />
            <span>{children}</span>
          </li>
        ),
        code: ({ children }) => (
          <code className="text-sm font-mono bg-surface-4 text-accent-light px-1.5 py-0.5 rounded border border-border">
            {children}
          </code>
        ),
        hr: () => <hr className="border-border my-5" />,
        blockquote: ({ children }) => (
          <blockquote className="border-l-2 border-accent/40 pl-4 my-3 text-gray-400 italic">
            {children}
          </blockquote>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

/* ── Expander ──────────────────────────────────────────────────────────────── */
interface ExpanderProps {
  title: string;
  icon: React.ReactNode;
  count: number;
  children: React.ReactNode;
  accent?: boolean;
}

function Expander({ title, icon, count, children, accent }: ExpanderProps) {
  const [open, setOpen] = useState(false);
  return (
    <div className={`border rounded-xl overflow-hidden transition-all duration-200
      ${accent ? "border-accent/25 shadow-[0_0_0_1px_rgba(124,58,237,0.08)]" : "border-border"}`}>
      <button
        onClick={() => setOpen((v) => !v)}
        className={`w-full flex items-center justify-between px-5 py-3.5 transition-colors
          ${accent ? "bg-accent/10 hover:bg-accent/15" : "bg-surface-2 hover:bg-surface-3"}`}
      >
        <div className="flex items-center gap-2.5 text-[15px] font-semibold text-gray-200">
          {icon}
          {title}
          <span className={`text-xs px-2 py-0.5 rounded-full font-semibold
            ${accent ? "bg-accent/25 text-accent-light" : "bg-surface-4 text-gray-400"}`}>
            {count}
          </span>
        </div>
        <span className="text-gray-500">
          {open ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
        </span>
      </button>
      {open && (
        <div className="divide-y divide-border animate-in">{children}</div>
      )}
    </div>
  );
}

/* ── Timing chips ──────────────────────────────────────────────────────────── */
function TimingChips({ result }: { result: AnalyzeResult }) {
  const chips = [
    { label: `${result.counts.tickets} similar tickets`, time: result.timings.tickets_s, icon: <Ticket size={11} /> },
    { label: `${result.counts.kb} KB articles`,          time: result.timings.kb_s,      icon: <BookOpen size={11} /> },
    { label: `LLM response`,                             time: result.timings.llm_s,     icon: <Sparkles size={11} /> },
  ];
  return (
    <div className="flex flex-wrap items-center gap-2">
      {chips.map((c) => (
        <div key={c.label}
          className="flex items-center gap-1.5 text-sm text-gray-400 bg-surface-2 border border-border rounded-lg px-3 py-2">
          <span className="text-accent-light">{c.icon}</span>
          {c.label}
          <span className="text-gray-600 ml-0.5">· {c.time}s</span>
        </div>
      ))}
      <div className="ml-auto flex items-center gap-1.5 text-sm font-semibold text-green-400
        bg-green-500/10 border border-green-500/20 rounded-lg px-3 py-2">
        <Clock size={11} />
        Total: {result.timings.total_s}s
      </div>
    </div>
  );
}

/* ── Expandable ticket item ────────────────────────────────────────────────── */
function TicketItem({ t }: { t: AnalyzeResult["similar_tickets"][0] }) {
  const [open, setOpen] = useState(false);
  return (
    <button
      onClick={() => setOpen((v) => !v)}
      className="w-full text-left px-5 py-4 bg-surface-1 hover:bg-surface-2 transition-colors cursor-pointer"
    >
      <div className="flex items-start justify-between gap-2 mb-2">
        <p className="text-[15px] font-semibold text-gray-200 leading-snug">{t.title}</p>
        <div className="flex items-center gap-1.5 flex-shrink-0">
          <span className="text-xs font-mono text-accent-light bg-accent/15 border border-accent/20 px-1.5 py-0.5 rounded">
            {t.score.toFixed(3)}
          </span>
          <span className={`text-gray-500 transition-transform duration-200 ${open ? "rotate-90" : ""}`}>
            <ChevronRight size={14} />
          </span>
        </div>
      </div>

      <div className="flex gap-1.5 mb-2 flex-wrap">
        {t.category !== "N/A" && (
          <span className="text-xs bg-surface-3 text-gray-400 border border-border px-2 py-0.5 rounded">
            {t.category}
          </span>
        )}
        {t.component !== "N/A" && (
          <span className="text-xs bg-surface-3 text-gray-400 border border-border px-2 py-0.5 rounded">
            {t.component}
          </span>
        )}
        {t.severity !== "N/A" && (
          <span className="text-xs bg-surface-3 text-gray-400 border border-border px-2 py-0.5 rounded">
            {t.severity}
          </span>
        )}
        {t.status && t.status !== "N/A" && (
          <span className={`text-xs px-2 py-0.5 rounded font-medium border ${
            t.status === "Resolved" || t.status === "Closed"
              ? "bg-green-500/10 text-green-400 border-green-500/25"
              : "bg-yellow-500/10 text-yellow-400 border-yellow-500/25"
          }`}>
            {t.status}
          </span>
        )}
      </div>

      <p className={`text-sm text-gray-500 leading-relaxed transition-all duration-300 ${open ? "" : "line-clamp-2"}`}>
        {t.text}
      </p>

      {!open && (
        <span className="text-xs text-accent-light/70 mt-1 inline-block">Click to expand ↓</span>
      )}
    </button>
  );
}

/* ── Expandable KB article item ────────────────────────────────────────────── */
function KbItem({ a }: { a: AnalyzeResult["kb_articles"][0] }) {
  const [open, setOpen] = useState(false);
  return (
    <button
      onClick={() => setOpen((v) => !v)}
      className="w-full text-left px-5 py-4 bg-surface-1 hover:bg-surface-2 transition-colors cursor-pointer"
    >
      <div className="flex items-center justify-between gap-2 mb-2">
        <p className="text-sm font-semibold text-gray-300 flex items-center gap-1.5">
          <FileText size={11} className="text-gray-500" />
          {a.source_file}
        </p>
        <span className={`text-gray-500 flex-shrink-0 transition-transform duration-200 ${open ? "rotate-90" : ""}`}>
          <ChevronRight size={14} />
        </span>
      </div>

      <p className={`text-sm text-gray-500 leading-relaxed transition-all duration-300 ${open ? "" : "line-clamp-3"}`}>
        {a.text}
      </p>

      {!open && (
        <span className="text-xs text-gray-600 mt-1 inline-block">Click to expand ↓</span>
      )}
    </button>
  );
}

/* ── Blueprint card with collapse/expand ──────────────────────────────────── */
const PREVIEW_LINES = 6; // collapsed height in lines

function BlueprintCard({ resolution }: { resolution: string }) {
  const [expanded, setExpanded] = useState(false);
  const bodyRef = useRef<HTMLDivElement>(null);
  const [contentHeight, setContentHeight] = useState(0);

  useEffect(() => {
    if (bodyRef.current) setContentHeight(bodyRef.current.scrollHeight);
  }, [resolution]);

  const collapsedPx = PREVIEW_LINES * 24; // ~24px per line

  return (
    <div className="border border-accent/25 rounded-xl overflow-hidden shadow-[0_0_24px_rgba(124,58,237,0.07)]">
      {/* Header — clicking it also toggles */}
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center justify-between px-5 py-4
          bg-gradient-to-r from-accent/15 to-transparent border-b border-accent/20
          hover:from-accent/20 transition-all duration-200 group"
      >
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-md bg-accent/25 flex items-center justify-center">
            <FileText size={14} className="text-accent-light" />
          </div>
          <h3 className="text-base font-bold text-gray-100">Resolution Blueprint</h3>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-accent-light/60 group-hover:text-accent-light transition-colors">
            {expanded ? "Collapse" : "Expand all"}
          </span>
          <span className={`text-gray-400 transition-transform duration-300 ${expanded ? "rotate-180" : ""}`}>
            <ChevronDown size={16} />
          </span>
        </div>
      </button>

      {/* Body with animated height */}
      <div
        className="bg-surface-1 overflow-hidden transition-all duration-500 ease-in-out"
        style={{ maxHeight: expanded ? `${contentHeight + 80}px` : `${collapsedPx}px` }}
      >
        <div ref={bodyRef} className="px-6 py-6">
          {resolution ? (
            <MarkdownBody content={resolution} />
          ) : (
            <div className="flex items-center gap-2 text-sm text-yellow-400 bg-yellow-500/8
              border border-yellow-500/20 rounded-lg px-4 py-3">
              <span>⚠️</span>
              <span>The model returned an empty response. Try submitting again.</span>
            </div>
          )}
        </div>
      </div>

      {/* Expand/collapse footer */}
      <button
        onClick={() => setExpanded((v) => !v)}
        className={`w-full flex items-center justify-center gap-2 py-2.5 text-sm font-medium
          border-t transition-all duration-200
          ${expanded
            ? "border-border text-gray-500 hover:text-gray-300 bg-surface-1 hover:bg-surface-2"
            : "border-accent/20 text-accent-light bg-accent/5 hover:bg-accent/10"
          }`}
      >
        {expanded ? (
          <><ChevronUp size={14} /> Collapse</>
        ) : (
          <><ChevronDown size={14} /> Show full resolution</>
        )}
      </button>
    </div>
  );
}

/* ── Main component ────────────────────────────────────────────────────────── */
export default function ResultPanel({ result }: { result: AnalyzeResult }) {
  return (
    <div className="space-y-4">
      {/* Timing */}
      <TimingChips result={result} />

      {/* Resolution Blueprint */}
      <BlueprintCard resolution={result.resolution} />

      {/* Bottom row */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <Expander
          title="Similar Tickets"
          icon={<Ticket size={14} className="text-accent-light" />}
          count={result.similar_tickets.length}
          accent
        >
          {result.similar_tickets.map((t, i) => (
            <TicketItem key={i} t={t} />
          ))}
        </Expander>

        <Expander
          title="Knowledge Base Articles"
          icon={<BookOpen size={14} className="text-gray-400" />}
          count={result.kb_articles.length}
        >
          {result.kb_articles.map((a, i) => (
            <KbItem key={i} a={a} />
          ))}
        </Expander>
      </div>
    </div>
  );
}
