import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  Bot,
  CheckCircle2,
  Headphones,
  Loader2,
  LogOut,
  Mic,
  MicOff,
  RefreshCcw,
  Send,
  Sparkles,
  UserRound,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import {
  createSession,
  getSession,
  postAction,
  postMessage,
  type ChatMessage,
  type SessionState,
} from "../api";
import NeuralBg from "../components/NeuralBg";
import CursorGlow from "../components/CursorGlow";
import { useSpeechRecognition } from "../hooks/useSpeechRecognition";

/* ── UI bits ────────────────────────────────────────────────────────────── */

function SoundWave() {
  return (
    <div className="flex items-end gap-0.5 h-4">
      {[1, 2, 3, 4, 3, 2, 1].map((h, i) => (
        <span
          key={i}
          className="w-0.5 rounded-full bg-red-400"
          style={{
            height: `${h * 3 + 3}px`,
            animation: "soundBar 0.8s ease-in-out infinite alternate",
            animationDelay: `${i * 0.1}s`,
          }}
        />
      ))}
    </div>
  );
}

function TypingDots() {
  return (
    <div className="flex items-center gap-1.5">
      <span className="w-1.5 h-1.5 rounded-full bg-accent-light typing-dot" />
      <span
        className="w-1.5 h-1.5 rounded-full bg-accent-light typing-dot"
        style={{ animationDelay: "0.2s" }}
      />
      <span
        className="w-1.5 h-1.5 rounded-full bg-accent-light typing-dot"
        style={{ animationDelay: "0.4s" }}
      />
    </div>
  );
}

function Avatar({ sender }: { sender: ChatMessage["sender"] }) {
  if (sender === "user") {
    return (
      <div className="shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-accent to-purple-700 text-white flex items-center justify-center shadow-lg shadow-accent/30">
        <UserRound size={16} />
      </div>
    );
  }
  if (sender === "ai") {
    return (
      <div className="shrink-0 w-8 h-8 rounded-full bg-surface-3 border border-accent/40 text-accent-light flex items-center justify-center">
        <Bot size={16} />
      </div>
    );
  }
  return (
    <div className="shrink-0 w-8 h-8 rounded-full bg-surface-3 border border-border text-gray-500 flex items-center justify-center">
      <Sparkles size={14} />
    </div>
  );
}

function Bubble({ m }: { m: ChatMessage }) {
  const isUser = m.sender === "user";
  const isAI = m.sender === "ai";
  const isSystem = m.sender === "system";

  const rowDir = isUser ? "flex-row-reverse" : "flex-row";

  const bubbleClass = isUser
    ? "bg-gradient-to-br from-accent to-purple-700 text-white border-accent/30 shadow-lg shadow-accent/20"
    : isAI
      ? "bg-surface-2/90 border-accent/20 text-gray-200"
      : "bg-surface-3/60 border-border/60 text-gray-400 italic";

  return (
    <div className={`flex items-start gap-2.5 ${rowDir} animate-bubble mb-4`}>
      <Avatar sender={m.sender} />
      <div
        className={`max-w-[86%] rounded-2xl px-4 py-2.5 border text-[15px] leading-relaxed ${bubbleClass}`}
        style={isUser ? { borderTopRightRadius: 6 } : { borderTopLeftRadius: 6 }}
      >
        {isAI || isSystem ? (
          <div className="prose prose-invert prose-sm max-w-none">
            <ReactMarkdown>{m.content}</ReactMarkdown>
          </div>
        ) : (
          <p className="whitespace-pre-wrap">{m.content}</p>
        )}
      </div>
    </div>
  );
}

function TicketCard({ ticket }: { ticket: NonNullable<SessionState["ticket"]> }) {
  return (
    <div className="mt-4 shimmer-border rounded-2xl bg-surface-2/90 backdrop-blur p-6 text-center">
      <div className="inline-flex items-center gap-2 text-xs uppercase tracking-widest text-accent-light mb-2">
        <CheckCircle2 size={14} /> Ticket Created
      </div>
      <p className="font-mono text-2xl font-bold bg-gradient-to-r from-accent-light via-cyan-300 to-accent bg-clip-text text-transparent">
        {ticket.public_id}
      </p>
      <p className="mt-3 text-sm text-gray-300 whitespace-pre-wrap max-w-md mx-auto">
        {ticket.issue_summary}
      </p>
      <div className="mt-4 flex flex-wrap justify-center gap-2 text-xs">
        <span className="px-3 py-1 rounded-full bg-accent/15 border border-accent/30 text-accent-light">
          Routed to · {ticket.department}
          {typeof ticket.department_confidence === "number"
            ? ` · ${Math.round(ticket.department_confidence * 100)}%`
            : ""}
        </span>
        <span className="px-3 py-1 rounded-full bg-surface-3 border border-border text-gray-400">
          user_id · {ticket.user_id}
        </span>
        <span className="px-3 py-1 rounded-full bg-surface-3 border border-border text-gray-400 capitalize">
          Status · {ticket.status}
        </span>
      </div>
      {ticket.routing_reason ? (
        <p className="mt-3 text-xs text-gray-500 max-w-md mx-auto">
          Routing note: {ticket.routing_reason}
        </p>
      ) : null}
    </div>
  );
}

/* ── Page ───────────────────────────────────────────────────────────────── */

export default function UserSupportPage() {
  const nav = useNavigate();
  const [session, setSession] = useState<SessionState | null>(null);
  const [input, setInput] = useState("");
  const [interim, setInterim] = useState("");
  const [sending, setSending] = useState(false);
  const [err, setErr] = useState("");
  const committed = useRef("");
  const endRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);

  const me = (() => {
    try {
      return JSON.parse(localStorage.getItem("its_user") || "null") as
        | { id: number; email: string; name: string }
        | null;
    } catch {
      return null;
    }
  })();

  const { listening, supported, start, stop } = useSpeechRecognition({
    onFinal: (t) => {
      const a = (committed.current + (committed.current ? " " : "") + t.trim()).trimStart();
      committed.current = a;
      setInput(a);
      setInterim("");
    },
    onInterim: setInterim,
    onError: (msg) => {
      setErr(msg);
      setTimeout(() => setErr(""), 5000);
    },
  });

  const syncInput = (v: string) => {
    committed.current = v;
    setInput(v);
    setInterim("");
  };

  const load = useCallback(async (sid: string) => {
    const s = await getSession(sid);
    setSession(s);
    requestAnimationFrame(() => endRef.current?.scrollIntoView({ behavior: "smooth" }));
  }, []);

  const didInit = useRef(false);
  useEffect(() => {
    if (didInit.current) return;
    didInit.current = true;
    (async () => {
      if (!localStorage.getItem("its_token")) {
        nav("/login?role=user", { replace: true });
        return;
      }
      try {
        const c = await createSession();
        await load(c.session_id);
      } catch (e) {
        setErr(e instanceof Error ? e.message : "Session error");
      } finally {
        setLoading(false);
      }
    })();
  }, [load, nav]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [session?.messages.length, sending]);

  async function send() {
    if (!session || sending) return;
    if (session.needs_escalation_choice) {
      setErr("Use the buttons: Try again or Connect to specialist.");
      return;
    }
    if (session.state === "escalated") return;
    const t = (input + (interim ? (input ? " " : "") + interim : "")).trim();
    if (!t) return;
    if (listening) stop();
    setSending(true);
    setErr("");
    syncInput("");

    // Optimistic user bubble
    const optimistic: ChatMessage = {
      id: -Date.now(),
      sender: "user",
      content: t,
      created_at: new Date().toISOString(),
    };
    setSession({ ...session, messages: [...session.messages, optimistic] });

    try {
      await postMessage(session.session_id, t);
      await load(session.session_id);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Failed");
    } finally {
      setSending(false);
    }
  }

  async function act(action: "retry" | "escalate") {
    if (!session) return;
    setSending(true);
    setErr("");
    try {
      await postAction(session.session_id, action);
      await load(session.session_id);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Action failed");
    } finally {
      setSending(false);
    }
  }

  function logout() {
    localStorage.removeItem("its_token");
    localStorage.removeItem("its_user");
    nav("/", { replace: true });
  }

  const display = input + (interim ? (input ? " " : "") + interim : "");

  if (loading) {
    return (
      <>
        <NeuralBg />
        <div
          className="min-h-screen flex items-center justify-center text-gray-400 bg-surface-0/80"
          style={{ cursor: "auto" }}
        >
          <div className="flex items-center gap-3">
            <Loader2 className="animate-spin text-accent-light" size={18} />
            Starting your support session…
          </div>
        </div>
      </>
    );
  }

  const canSend =
    !sending && !!display.trim() && !session?.needs_escalation_choice && session?.state !== "escalated";

  return (
    <>
      <NeuralBg />
      <CursorGlow />
      <div
        className="relative z-10 min-h-screen flex flex-col bg-surface-0/85 text-gray-200"
        style={{ cursor: "none" }}
      >
        {/* Header */}
        <header className="sticky top-0 z-20 backdrop-blur-md bg-surface-0/60 border-b border-border">
          <div className="max-w-3xl mx-auto w-full flex items-center justify-between px-5 py-3">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-accent to-purple-700 flex items-center justify-center shadow-lg shadow-accent/30">
                <Headphones size={18} className="text-white" />
              </div>
              <div>
                <h1 className="text-sm font-bold text-gray-100 leading-tight">Support Chat</h1>
                <p className="text-[11px] text-gray-500">
                  Signed in as <span className="text-accent-light">{me?.email ?? "demo user"}</span>{" "}
                  · user_id {me?.id ?? "?"}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Link
                to="/"
                className="text-xs text-gray-400 hover:text-accent-light"
                style={{ cursor: "none" }}
              >
                Home
              </Link>
              <button
                type="button"
                onClick={logout}
                style={{ cursor: "none" }}
                className="text-xs text-gray-400 hover:text-red-300 flex items-center gap-1"
              >
                <LogOut size={12} /> Log out
              </button>
            </div>
          </div>
        </header>

        {/* Messages */}
        <main className="flex-1 overflow-y-auto">
          <div className="max-w-3xl mx-auto w-full px-4 md:px-6 py-6">
            {/* Welcome hero (only before any user message) */}
            {session &&
              session.messages.filter((m) => m.sender === "user").length === 0 && (
                <div className="text-center py-8 select-none">
                  <div className="inline-flex items-center gap-2 text-[11px] font-medium uppercase tracking-widest text-accent-light/80 border border-accent/20 bg-accent/5 px-3 py-1 rounded-full mb-4">
                    <Sparkles size={12} /> AI Helpdesk
                  </div>
                  <h2 className="text-2xl md:text-3xl font-bold text-gray-100 tracking-tight">
                    How can we{" "}
                    <span className="bg-gradient-to-r from-accent-light via-cyan-300 to-accent bg-clip-text text-transparent">
                      help you today?
                    </span>
                  </h2>
                  <p className="text-sm text-gray-500 max-w-md mx-auto mt-2">
                    Describe your issue — type or use voice. Our AI tries twice; if it still can’t
                    fix it, we’ll create a specialist ticket for you.
                  </p>
                </div>
              )}

            {err && (
              <div className="mb-3 mx-auto max-w-md text-center text-xs bg-yellow-500/10 border border-yellow-500/30 text-yellow-200 px-3 py-2 rounded-xl">
                {err}
              </div>
            )}

            {session?.messages.map((m) => (
              <Bubble key={m.id} m={m} />
            ))}

            {sending && (
              <div className="flex items-start gap-2.5 animate-bubble mb-4">
                <Avatar sender="ai" />
                <div className="max-w-[60%] rounded-2xl px-4 py-3 bg-surface-2/80 border border-accent/20">
                  <TypingDots />
                </div>
              </div>
            )}

            {session?.ticket && <TicketCard ticket={session.ticket} />}

            <div ref={endRef} />
          </div>
        </main>

        {/* Composer */}
        {session && session.state === "active" && (
          <footer className="sticky bottom-0 z-20 backdrop-blur-md bg-surface-0/80 border-t border-border">
            <div className="max-w-3xl mx-auto w-full px-4 md:px-6 py-4">
              {session.needs_escalation_choice && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-3">
                  <button
                    type="button"
                    disabled={sending}
                    onClick={() => act("retry")}
                    style={{ cursor: sending ? "not-allowed" : "none" }}
                    className="group flex items-center justify-center gap-2 py-3 rounded-xl border border-border bg-surface-2 hover:border-accent/40 hover:bg-surface-3 text-sm font-medium text-gray-200 transition-all disabled:opacity-50"
                  >
                    <RefreshCcw size={14} className="text-accent-light" />
                    Try AI again
                  </button>
                  <button
                    type="button"
                    disabled={sending}
                    onClick={() => act("escalate")}
                    style={{ cursor: sending ? "not-allowed" : "none" }}
                    className="flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-semibold text-white bg-gradient-to-r from-accent to-purple-700 hover:from-accent-hover hover:to-purple-800 shadow-lg shadow-accent/25 hover:scale-[1.01] active:scale-[0.99] transition-all disabled:opacity-50"
                  >
                    <Headphones size={14} />
                    Connect to specialist
                  </button>
                </div>
              )}

              <div
                className={`relative flex items-end gap-2 bg-surface-2/80 border rounded-2xl p-2 shadow-[0_2px_24px_rgba(0,0,0,0.35)] transition-colors ${
                  listening ? "border-red-500/40" : "border-border focus-within:border-accent/50"
                }`}
              >
                {supported && (
                  <button
                    type="button"
                    onClick={() =>
                      listening ? (stop(), setInterim("")) : (setErr(""), start())
                    }
                    title={listening ? "Stop voice" : "Start voice"}
                    style={{ cursor: "none" }}
                    className={`shrink-0 w-11 h-11 rounded-xl border flex items-center justify-center transition-all ${
                      listening
                        ? "bg-red-500/15 border-red-500/40 text-red-400 mic-pulse"
                        : "bg-surface-3 border-border text-gray-400 hover:text-accent-light hover:border-accent/40"
                    }`}
                  >
                    {listening ? <MicOff size={16} /> : <Mic size={16} />}
                  </button>
                )}

                <textarea
                  rows={1}
                  className="flex-1 bg-transparent text-[15px] text-gray-200 placeholder-gray-600 px-2 py-2.5 resize-none max-h-40 focus:outline-none"
                  placeholder={
                    session.needs_escalation_choice
                      ? "Pick an option above to continue…"
                      : listening
                        ? "Listening… speak now"
                        : "Describe your issue — or press the mic to speak"
                  }
                  value={display}
                  disabled={!!session.needs_escalation_choice}
                  onChange={(e) => syncInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      void send();
                    }
                  }}
                  style={{ cursor: "text" }}
                />

                {listening && (
                  <div className="self-center mr-1 px-2 py-1 rounded-full bg-red-500/15 border border-red-500/30 text-[10px] font-semibold text-red-300 flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse" /> REC
                    <SoundWave />
                  </div>
                )}

                <button
                  type="button"
                  disabled={!canSend}
                  onClick={send}
                  title="Send (Enter)"
                  style={{ cursor: canSend ? "none" : "not-allowed" }}
                  className="shrink-0 w-11 h-11 rounded-xl flex items-center justify-center text-white bg-gradient-to-br from-accent to-purple-700 hover:from-accent-hover hover:to-purple-800 shadow-lg shadow-accent/30 disabled:opacity-40 disabled:saturate-50 transition-all hover:scale-[1.03] active:scale-[0.97]"
                >
                  {sending ? <Loader2 className="animate-spin" size={16} /> : <Send size={16} />}
                </button>
              </div>
              <p className="text-[10px] text-gray-600 mt-1.5 pl-1">
                Enter to send · Shift+Enter for a new line · up to 2 AI tries before specialist
              </p>
            </div>
          </footer>
        )}

        {session?.state === "escalated" && (
          <footer className="sticky bottom-0 z-20 py-3 text-center text-xs text-gray-500 border-t border-border bg-surface-0/80 backdrop-blur">
            Session complete — a specialist will pick up this ticket.
          </footer>
        )}
      </div>
    </>
  );
}
