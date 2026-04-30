import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  Bot,
  Copy,
  Headphones,
  Inbox,
  Loader2,
  LogOut,
  Mail,
  Plus,
  RefreshCcw,
  RotateCcw,
  Sparkles,
  Ticket,
  User as UserIcon,
  X,
} from "lucide-react";
import {
  adminCreateTicket,
  generateAdminInsights,
  getCachedAdminInsights,
  getAdminTicket,
  listAdminTickets,
  refreshAdminInsights,
  type AdminDetail,
  type AdminTicketRow,
  type ChatMessage,
  type CreateTicketRes,
  type Insights,
} from "../api";
import NeuralBg from "../components/NeuralBg";
import ResultPanel from "../components/ResultPanel";
import type { AnalyzeResult } from "../types";

function insightsToResult(ins: Insights): AnalyzeResult {
  return {
    resolution: ins.resolution,
    similar_tickets: (ins.similar_tickets || []) as AnalyzeResult["similar_tickets"],
    kb_articles: (ins.kb_articles || []) as AnalyzeResult["kb_articles"],
    timings: {
      tickets_s: ins.timings?.tickets_s ?? 0,
      kb_s: ins.timings?.kb_s ?? 0,
      llm_s: ins.timings?.llm_s ?? 0,
      total_s: ins.timings?.total_s ?? 0,
    },
    counts: { tickets: 0, kb: 0 },
  };
}

function StatusDot({ status }: { status: string }) {
  const s = status.toLowerCase();
  const cls =
    s === "open"
      ? "bg-emerald-400"
      : s === "closed" || s === "resolved"
        ? "bg-gray-500"
        : "bg-amber-400";
  return <span className={`inline-block w-1.5 h-1.5 rounded-full ${cls}`} />;
}

function DepartmentTag({ dept }: { dept: string }) {
  const palette: Record<string, string> = {
    billing: "bg-cyan-500/10 border-cyan-500/30 text-cyan-300",
    payments: "bg-cyan-500/10 border-cyan-500/30 text-cyan-300",
    technical: "bg-accent/15 border-accent/30 text-accent-light",
    tech: "bg-accent/15 border-accent/30 text-accent-light",
    account: "bg-amber-500/10 border-amber-500/30 text-amber-200",
    general: "bg-surface-3 border-border text-gray-300",
  };
  const k = Object.keys(palette).find((w) => dept.toLowerCase().includes(w)) || "general";
  return (
    <span
      className={`px-2 py-0.5 rounded-full border text-[10px] font-semibold uppercase tracking-wider ${palette[k]}`}
    >
      {dept}
    </span>
  );
}

function ChatHistory({ messages }: { messages: ChatMessage[] }) {
  if (messages.length === 0) {
    return <p className="text-xs text-gray-500">No messages yet.</p>;
  }
  return (
    <ul className="space-y-2">
      {messages.map((m) => {
        const isUser = m.sender === "user";
        const isAI = m.sender === "ai";
        return (
          <li
            key={m.id}
            className={`text-[12px] leading-relaxed rounded-lg px-3 py-2 border ${
              isUser
                ? "bg-accent/10 border-accent/20 text-gray-200"
                : isAI
                  ? "bg-surface-2 border-border text-gray-300"
                  : "bg-surface-3/50 border-border/60 text-gray-500 italic"
            }`}
          >
            <div className="flex items-center gap-1.5 mb-0.5">
              {isUser ? (
                <UserIcon size={11} className="text-accent-light" />
              ) : isAI ? (
                <Bot size={11} className="text-accent-light" />
              ) : (
                <Sparkles size={11} className="text-gray-500" />
              )}
              <span className="text-[10px] uppercase tracking-wider text-gray-500">
                {m.sender}
              </span>
            </div>
            <p className="whitespace-pre-wrap">{m.content}</p>
          </li>
        );
      })}
    </ul>
  );
}

const DEPARTMENTS = [
  "IT Infrastructure & Platform",
  "End-User & Desktop Support",
  "Network & Security",
  "Applications & Data Services",
];

function NewTicketModal({
  onClose,
  onCreated,
}: {
  onClose: () => void;
  onCreated: (res: CreateTicketRes) => void;
}) {
  const [desc, setDesc] = useState("");
  const [dept, setDept] = useState("");
  const [customRes, setCustomRes] = useState("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [created, setCreated] = useState<CreateTicketRes | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (desc.trim().length < 10) {
      setErr("Issue description must be at least 10 characters.");
      return;
    }
    setErr("");
    setLoading(true);
    try {
      const res = await adminCreateTicket(desc.trim(), dept, customRes.trim());
      setCreated(res);
      onCreated(res);
    } catch (ex) {
      setErr(ex instanceof Error ? ex.message : "Failed to create ticket");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <div className="relative w-full max-w-2xl bg-surface-1 border border-border rounded-2xl shadow-2xl shadow-black/60 flex flex-col max-h-[90vh]">
        {/* Modal header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-border shrink-0">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-accent to-purple-700 flex items-center justify-center">
              <Plus size={14} className="text-white" />
            </div>
            <h2 className="text-sm font-bold text-gray-100">New Ticket</h2>
            <span className="text-[10px] text-gray-500 border border-border rounded-full px-2 py-0.5">
              Admin — AI Resolution
            </span>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-gray-500 hover:text-gray-300 transition-colors"
          >
            <X size={16} />
          </button>
        </div>

        {created ? (
          /* ── Success state ── */
          <div className="flex-1 overflow-y-auto p-5 space-y-4">
            <div className="flex items-center gap-2 text-emerald-400 text-sm font-semibold">
              <span className="w-5 h-5 rounded-full bg-emerald-400/20 border border-emerald-400/40 flex items-center justify-center text-[10px]">✓</span>
              Ticket created & indexed in retrieval database
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-surface-2/60 border border-border rounded-xl p-3">
                <p className="text-[10px] uppercase tracking-widest text-gray-500 mb-1">Ticket ID</p>
                <p className="font-mono text-sm font-bold text-accent-light">{created.ticket.public_id}</p>
              </div>
              <div className="bg-surface-2/60 border border-border rounded-xl p-3">
                <p className="text-[10px] uppercase tracking-widest text-gray-500 mb-1">Department</p>
                <p className="text-sm text-gray-200">
                  {created.ticket.department}
                  {typeof created.ticket.department_confidence === "number"
                    ? ` (${Math.round(created.ticket.department_confidence * 100)}%)`
                    : ""}
                </p>
              </div>
            </div>
            {created.ticket.routing_reason ? (
              <div className="bg-surface-2/40 border border-border rounded-xl p-3">
                <p className="text-[10px] uppercase tracking-widest text-gray-500 mb-1">Routing Note</p>
                <p className="text-xs text-gray-400">{created.ticket.routing_reason}</p>
              </div>
            ) : null}
            <div className="bg-surface-2/40 border border-border rounded-xl p-3">
              <p className="text-[10px] uppercase tracking-widest text-gray-500 mb-2">Generated Resolution (saved to DB)</p>
              <pre className="text-[11px] text-gray-300 whitespace-pre-wrap leading-relaxed max-h-64 overflow-y-auto">
                {created.ticket.resolution}
              </pre>
            </div>
            <p className="text-[11px] text-gray-500">
              This ticket and its resolution are now indexed — future similar issues will retrieve this resolution.
            </p>
            <div className="flex gap-2 pt-1">
              <button
                type="button"
                onClick={() => { setCreated(null); setDesc(""); setDept(""); setCustomRes(""); }}
                className="flex-1 py-2 text-xs rounded-xl border border-border text-gray-300 hover:border-accent/40 transition-colors"
              >
                Create Another
              </button>
              <button
                type="button"
                onClick={onClose}
                className="flex-1 py-2 text-xs rounded-xl bg-accent/20 border border-accent/30 text-accent-light hover:bg-accent/30 transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        ) : (
          /* ── Form state ── */
          <form onSubmit={(e) => void handleSubmit(e)} className="flex-1 overflow-y-auto p-5 space-y-4">
            <div>
              <label className="block text-[10px] uppercase tracking-widest text-gray-500 mb-1.5">
                Issue Description <span className="text-red-400">*</span>
              </label>
              <textarea
                ref={textareaRef}
                value={desc}
                onChange={(e) => setDesc(e.target.value)}
                rows={5}
                placeholder="Describe the IT issue in detail — e.g., 'VPN keeps disconnecting after 10 minutes on Windows 11 laptops in the Chicago office…'"
                className="w-full bg-surface-0 border border-border rounded-xl px-3 py-2.5 text-[13px] text-gray-200 placeholder-gray-600 focus:border-accent/50 focus:outline-none resize-none leading-relaxed"
                disabled={loading}
              />
            </div>

            <div>
              <label className="block text-[10px] uppercase tracking-widest text-gray-500 mb-1.5">
                Department <span className="text-gray-600">(auto-detected if blank)</span>
              </label>
              <select
                value={dept}
                onChange={(e) => setDept(e.target.value)}
                className="w-full bg-surface-0 border border-border rounded-xl px-3 py-2.5 text-[13px] text-gray-200 focus:border-accent/50 focus:outline-none"
                disabled={loading}
              >
                <option value="">— Auto-detect —</option>
                {DEPARTMENTS.map((d) => (
                  <option key={d} value={d}>{d}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-[10px] uppercase tracking-widest text-gray-500 mb-1.5">
                Custom Resolution <span className="text-gray-600">(optional — leave blank for AI generation)</span>
              </label>
              <textarea
                value={customRes}
                onChange={(e) => setCustomRes(e.target.value)}
                rows={3}
                placeholder="Paste a known resolution here, or leave blank to let the AI generate one using RAG…"
                className="w-full bg-surface-0 border border-border rounded-xl px-3 py-2.5 text-[13px] text-gray-200 placeholder-gray-600 focus:border-accent/50 focus:outline-none resize-none leading-relaxed"
                disabled={loading}
              />
            </div>

            {err && (
              <div className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
                {err}
              </div>
            )}

            <div className="flex items-center gap-2 pt-1 shrink-0">
              <button
                type="button"
                onClick={onClose}
                disabled={loading}
                className="flex-1 py-2.5 text-xs rounded-xl border border-border text-gray-400 hover:text-gray-200 transition-colors disabled:opacity-40"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={loading || desc.trim().length < 10}
                className="flex-[2] py-2.5 text-xs font-semibold rounded-xl bg-gradient-to-r from-accent to-purple-700 text-white shadow-md shadow-accent/30 hover:opacity-90 transition-opacity disabled:opacity-40 flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <Loader2 size={13} className="animate-spin" />
                    Generating resolution & indexing…
                  </>
                ) : (
                  <>
                    <Sparkles size={13} />
                    {customRes.trim() ? "Save Ticket & Index" : "Generate Resolution & Create Ticket"}
                  </>
                )}
              </button>
            </div>
            {!customRes.trim() && (
              <p className="text-[10px] text-gray-600 text-center -mt-2">
                AI will run RAG retrieval + LLM to generate the resolution — this may take ~30s.
              </p>
            )}
          </form>
        )}
      </div>
    </div>
  );
}

export default function AdminPage() {
  const nav = useNavigate();
  const [rows, setRows] = useState<AdminTicketRow[]>([]);
  const [sel, setSel] = useState<AdminDetail | null>(null);
  const [ins, setIns] = useState<Insights | null>(null);
  // Component-level cache: ticketId → Insights. Avoids re-fetching the same ticket.
  // Backed by the DB cache, so a page-refresh also skips the LLM call.
  const [insCache, setInsCache] = useState<Map<number, Insights>>(new Map());
  const [listErr, setListErr] = useState("");
  const [insLoading, setInsLoading] = useState(false);
  const [insCheckingCache, setInsCheckingCache] = useState(false);
  const [insRefreshing, setInsRefreshing] = useState(false);
  const [filter, setFilter] = useState("");
  const [refreshing, setRefreshing] = useState(false);
  const [copied, setCopied] = useState(false);
  const [showNewTicket, setShowNewTicket] = useState(false);

  const me = (() => {
    try {
      return JSON.parse(localStorage.getItem("its_user") || "null") as
        | { id: number; email: string; name: string; role: string }
        | null;
    } catch {
      return null;
    }
  })();

  const loadList = useCallback(async () => {
    setRefreshing(true);
    try {
      const r = await listAdminTickets();
      setRows(r.tickets);
      setListErr("");
    } catch (e) {
      setListErr(e instanceof Error ? e.message : "List failed");
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    if (!localStorage.getItem("its_token")) {
      nav("/login?role=admin", { replace: true });
      return;
    }
    const u = JSON.parse(localStorage.getItem("its_user") || "null");
    if (u?.role !== "admin") {
      nav("/", { replace: true });
      return;
    }
    void loadList();
  }, [loadList, nav]);

  const openTicket = async (id: number) => {
    // Already on this ticket — nothing to do. Recommended actions only run
    // when the admin explicitly clicks Generate/Refresh.
    if (sel?.ticket.id === id) return;

    // Switching to a different ticket — clear stale insights from view.
    setIns(null);

    try {
      const d = await getAdminTicket(id);
      setSel(d);

      // Check component-level cache first (zero network cost).
      if (insCache.has(id)) {
        setIns(insCache.get(id)!);
        return;
      }

      // Cache-only lookup. This never runs retrieval or the LLM.
      setInsCheckingCache(true);
      const cached = await getCachedAdminInsights(id);
      if (cached) {
        setIns(cached);
        setInsCache((prev) => new Map(prev).set(id, cached));
      }
    } catch (e) {
      setListErr(e instanceof Error ? e.message : "Load failed");
    } finally {
      setInsCheckingCache(false);
    }
  };

  const handleGenerateInsights = async () => {
    if (!sel) return;
    setInsLoading(true);
    setIns(null);
    try {
      const r = await generateAdminInsights(sel.ticket.id);
      setIns(r);
      setInsCache((prev) => new Map(prev).set(sel.ticket.id, r));
    } catch (e) {
      setListErr(e instanceof Error ? e.message : "Generate failed");
    } finally {
      setInsLoading(false);
    }
  };

  const handleRefreshInsights = async () => {
    if (!sel) return;
    setInsRefreshing(true);
    setIns(null);
    try {
      const r = await refreshAdminInsights(sel.ticket.id);
      setIns(r);
      setInsCache((prev) => new Map(prev).set(sel.ticket.id, r));
    } catch (e) {
      setListErr(e instanceof Error ? e.message : "Refresh failed");
    } finally {
      setInsRefreshing(false);
    }
  };

  function logout() {
    localStorage.removeItem("its_token");
    localStorage.removeItem("its_user");
    nav("/", { replace: true });
  }

  async function copyEmail() {
    if (!sel) return;
    try {
      await navigator.clipboard.writeText(sel.fake_outbound_email.body);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* ignore */
    }
  }

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return rows;
    return rows.filter(
      (t) =>
        t.public_id.toLowerCase().includes(q) ||
        t.department.toLowerCase().includes(q) ||
        t.issue_excerpt.toLowerCase().includes(q) ||
        t.user_email?.toLowerCase().includes(q),
    );
  }, [rows, filter]);

  return (
    <>
      <NeuralBg />
      {showNewTicket && (
        <NewTicketModal
          onClose={() => setShowNewTicket(false)}
          onCreated={(res) => {
            setShowNewTicket(false);
            // Pre-populate the insights cache so clicking the new ticket is instant.
            if (res.insights) {
              setInsCache((prev) => new Map(prev).set(res.ticket.id, res.insights));
            }
            void loadList();
          }}
        />
      )}
      <div
        className="relative z-10 min-h-screen flex flex-col text-gray-200 bg-surface-0/90"
        style={{ cursor: "auto" }}
      >
        {/* Header */}
        <header className="sticky top-0 z-20 backdrop-blur-md bg-surface-0/60 border-b border-border">
          <div className="px-5 py-3 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-700 flex items-center justify-center shadow-lg shadow-cyan-500/30">
                <Headphones size={18} className="text-white" />
              </div>
              <div>
                <h1 className="text-sm font-bold text-gray-100 leading-tight">Specialist Workbench</h1>
                <p className="text-[11px] text-gray-500">
                  Signed in as <span className="text-cyan-300">{me?.email ?? "admin"}</span> ·{" "}
                  {rows.length} ticket{rows.length === 1 ? "" : "s"}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => setShowNewTicket(true)}
                className="flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-lg bg-accent/20 border border-accent/30 text-accent-light hover:bg-accent/30 transition-colors"
              >
                <Plus size={13} /> New Ticket
              </button>
              <button
                type="button"
                onClick={() => void loadList()}
                className="text-xs text-gray-400 hover:text-cyan-300 flex items-center gap-1"
              >
                <RefreshCcw size={12} className={refreshing ? "animate-spin" : ""} />
                Refresh
              </button>
              <Link to="/" className="text-xs text-gray-400 hover:text-accent-light">
                Home
              </Link>
              <button
                type="button"
                onClick={logout}
                className="text-xs text-gray-400 hover:text-red-300 flex items-center gap-1"
              >
                <LogOut size={12} /> Log out
              </button>
            </div>
          </div>
        </header>

        <div className="flex-1 grid grid-cols-1 md:grid-cols-12 gap-0 overflow-hidden">
          {/* Queue */}
          <aside className="md:col-span-3 border-b md:border-b-0 md:border-r border-border max-h-[calc(100vh-57px)] overflow-y-auto bg-surface-0/40">
            <div className="p-3 sticky top-0 bg-surface-0/80 backdrop-blur border-b border-border">
              <div className="flex items-center justify-between mb-2">
                <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-widest flex items-center gap-1.5">
                  <Inbox size={12} /> Queue
                </h2>
                <span className="text-[10px] text-gray-500">{filtered.length}</span>
              </div>
              <input
                type="text"
                placeholder="Search tickets…"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                className="w-full bg-surface-1 border border-border rounded-lg px-2.5 py-1.5 text-xs text-gray-300 placeholder-gray-600 focus:border-cyan-500/40"
              />
            </div>

            <ul className="p-2 space-y-1.5">
              {listErr && (
                <li className="text-red-400 text-xs bg-red-500/10 border border-red-500/20 rounded p-2">
                  {listErr}
                </li>
              )}
              {filtered.map((t) => {
                const active = sel?.ticket.id === t.id;
                return (
                  <li key={t.id}>
                    <button
                      type="button"
                      onClick={() => void openTicket(t.id)}
                      className={`w-full text-left p-2.5 rounded-xl border transition-all ${
                        active
                          ? "border-cyan-500/40 bg-cyan-500/10 shadow-md shadow-cyan-500/10"
                          : "border-border bg-surface-2/60 hover:border-accent/30 hover:bg-surface-2"
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-mono text-[12px] font-semibold text-accent-light">
                          {t.public_id}
                        </span>
                        <div className="flex items-center gap-1 text-[10px] text-gray-500 capitalize">
                          <StatusDot status={t.status} />
                          {t.status}
                        </div>
                      </div>
                      <p className="text-[11px] text-gray-400 line-clamp-2 leading-snug">
                        {t.issue_excerpt || "(no summary)"}
                      </p>
                      <div className="mt-1.5 flex items-center justify-between">
                        <DepartmentTag dept={t.department} />
                        <span className="text-[10px] text-gray-600">u#{t.user_id}</span>
                      </div>
                    </button>
                  </li>
                );
              })}
              {filtered.length === 0 && !listErr && (
                <li className="text-center text-xs text-gray-500 p-6 border border-dashed border-border rounded-xl">
                  {rows.length === 0
                    ? "No tickets yet. Create one from the user chat."
                    : "No tickets match your search."}
                </li>
              )}
            </ul>
          </aside>

          {/* Insights */}
          <section className="md:col-span-5 border-b md:border-b-0 md:border-r border-border max-h-[calc(100vh-57px)] overflow-y-auto">
            <div className="p-4">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-widest flex items-center gap-1.5">
                  <Sparkles size={12} className="text-accent-light" /> AI Insights
                </h2>
                {sel && (
                  <div className="flex items-center gap-2">
                    {!ins ? (
                      <button
                        type="button"
                        onClick={() => void handleGenerateInsights()}
                        disabled={insLoading || insRefreshing || insCheckingCache}
                        className="flex items-center gap-1.5 text-[10px] font-semibold px-2.5 py-1 rounded-lg bg-accent/15 border border-accent/25 text-accent-light hover:bg-accent/25 disabled:opacity-40 transition-colors"
                        title="Run RAG + LLM to generate recommended actions for this ticket"
                      >
                        <Sparkles size={11} className={insLoading ? "animate-pulse" : ""} />
                        {insLoading ? "Generating…" : "Generate Recommended Actions"}
                      </button>
                    ) : (
                      <button
                        type="button"
                        onClick={() => void handleRefreshInsights()}
                        disabled={insRefreshing || insLoading}
                        className="flex items-center gap-1 text-[10px] text-gray-500 hover:text-accent-light disabled:opacity-40 transition-colors"
                        title="Re-run RAG pipeline and regenerate insights"
                      >
                        <RotateCcw size={11} className={insRefreshing ? "animate-spin" : ""} />
                        {insRefreshing ? "Refreshing…" : "Refresh"}
                      </button>
                    )}
                  </div>
                )}
              </div>

              {!sel && (
                <div className="flex flex-col items-center justify-center text-center p-10 border border-dashed border-border rounded-2xl bg-surface-2/40">
                  <Sparkles className="text-accent-light mb-2" size={22} />
                  <p className="text-sm text-gray-300 font-medium">Pick a ticket to review</p>
                  <p className="text-xs text-gray-500 mt-1">
                    Similar tickets, KB articles, LLM-recommended steps and a draft email will
                    appear here.
                  </p>
                </div>
              )}

              {sel && insCheckingCache && !ins && (
                <div className="flex items-center gap-2 text-sm text-gray-400 bg-surface-2/60 border border-border rounded-xl p-6">
                  <Loader2 className="animate-spin text-accent-light" size={16} />
                  Checking saved recommended actions…
                </div>
              )}

              {sel && !ins && !insLoading && !insRefreshing && !insCheckingCache && (
                <div className="flex flex-col items-center justify-center text-center p-10 border border-dashed border-border rounded-2xl bg-surface-2/40">
                  <Sparkles className="text-accent-light mb-2" size={22} />
                  <p className="text-sm text-gray-300 font-medium">Recommended actions not generated yet</p>
                  <p className="text-xs text-gray-500 mt-1 max-w-sm">
                    Click <span className="text-accent-light">Generate Recommended Actions</span> to run
                    retrieval and the LLM for this ticket. Opening tickets no longer triggers AI automatically.
                  </p>
                </div>
              )}

              {sel && (insLoading || insRefreshing) && (
                <div className="flex items-center gap-2 text-sm text-gray-400 bg-surface-2/60 border border-border rounded-xl p-6">
                  <Loader2 className="animate-spin text-accent-light" size={16} />
                  {insRefreshing ? "Re-running retrieval + LLM…" : "Generating recommended actions…"}
                </div>
              )}

              {sel && ins && !insLoading && <ResultPanel result={insightsToResult(ins)} />}
            </div>
          </section>

          {/* Ticket details */}
          <section className="md:col-span-4 max-h-[calc(100vh-57px)] overflow-y-auto">
            <div className="p-4">
              {sel ? (
                <>
                  <div className="flex items-center gap-2">
                    <Ticket size={16} className="text-accent-light" />
                    <h2 className="font-mono text-base font-bold bg-gradient-to-r from-accent-light to-cyan-300 bg-clip-text text-transparent">
                      {sel.ticket.public_id}
                    </h2>
                    <DepartmentTag dept={sel.ticket.department} />
                  </div>
                  <p className="text-[11px] text-gray-500 mt-0.5">
                    DB id {sel.ticket.id} · session {sel.ticket.session_id.slice(0, 8)}…
                  </p>
                  {sel.ticket.routing_reason ? (
                    <div className="mt-2 text-[11px] text-gray-500">
                      Model 2 routing: {Math.round((sel.ticket.department_confidence ?? 0) * 100)}%
                      confidence · {sel.ticket.routing_reason}
                    </div>
                  ) : null}

                  <div className="mt-3 p-3 rounded-xl bg-surface-2/70 border border-border text-sm space-y-2">
                    <div className="flex items-center gap-2 text-gray-300">
                      <div className="w-7 h-7 rounded-full bg-gradient-to-br from-accent to-purple-700 flex items-center justify-center text-white">
                        <UserIcon size={13} />
                      </div>
                      <div className="leading-tight">
                        <p className="text-[13px] font-medium">
                          {sel.ticket.user_name || "Demo User"}
                        </p>
                        <p className="text-[11px] text-gray-500">
                          {sel.ticket.user_email || "n/a"} · user_id {sel.ticket.user_id}
                        </p>
                      </div>
                    </div>

                    <div className="pt-2 border-t border-border/60">
                      <p className="text-[10px] uppercase tracking-widest text-gray-500 mb-1">
                        Issue summary
                      </p>
                      <p className="whitespace-pre-wrap text-gray-300 text-[13px] leading-relaxed">
                        {sel.ticket.issue_summary}
                      </p>
                    </div>
                  </div>

                  <div className="mt-4">
                    <h3 className="text-[10px] uppercase tracking-widest text-gray-500 mb-2">
                      Full chat history ({sel.messages.length})
                    </h3>
                    <div className="max-h-64 overflow-y-auto pr-1">
                      <ChatHistory messages={sel.messages} />
                    </div>
                  </div>

                  <div className="mt-4">
                    <div className="flex items-center justify-between mb-1.5">
                      <h3 className="text-[10px] uppercase tracking-widest text-gray-500 flex items-center gap-1">
                        <Mail size={11} /> Draft email to user
                      </h3>
                      <button
                        type="button"
                        onClick={() => void copyEmail()}
                        className="text-[10px] text-accent-light hover:text-accent flex items-center gap-1"
                      >
                        <Copy size={10} /> {copied ? "Copied!" : "Copy"}
                      </button>
                    </div>
                    <div className="text-[11px] text-gray-500 mb-1">
                      To: <span className="text-gray-300">{sel.fake_outbound_email.to}</span> · Subject:{" "}
                      <span className="text-gray-300">{sel.fake_outbound_email.subject}</span>
                    </div>
                    <textarea
                      className="w-full h-40 text-[12px] font-mono bg-surface-1 border border-border rounded-lg p-2.5 text-gray-300 focus:border-accent/40"
                      defaultValue={sel.fake_outbound_email.body}
                    />
                    <p className="text-[10px] text-gray-600 mt-1">
                      Demo only — this email is not actually sent.
                    </p>
                  </div>
                </>
              ) : (
                <div className="text-center text-xs text-gray-500 p-6 border border-dashed border-border rounded-xl">
                  Select a ticket from the queue to see user info, chat history and the draft email.
                </div>
              )}
            </div>
          </section>
        </div>
      </div>
    </>
  );
}
