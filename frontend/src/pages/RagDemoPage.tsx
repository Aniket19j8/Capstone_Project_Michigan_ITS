import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2, Mic, MicOff, Send, Sparkles } from "lucide-react";
import { getStatus, analyze } from "../api";
import type { SystemStatus, AnalyzeResult } from "../types";
import TopBar from "../components/TopBar";
import ResultPanel from "../components/ResultPanel";
import CursorGlow from "../components/CursorGlow";
import NeuralBg from "../components/NeuralBg";
import { useSpeechRecognition } from "../hooks/useSpeechRecognition";
import { Link } from "react-router-dom";

type PipelineStep = false | "tickets" | "kb" | "llm";

const STEP_META = [
  { key: "tickets", label: "Retrieving Tickets" },
  { key: "kb", label: "Searching KB" },
  { key: "llm", label: "Generating Blueprint" },
] as const;

const FLOAT_CHIPS = [
  { text: "RAG Pipeline", delay: "0s" },
  { text: "Vector Search", delay: "0.6s" },
  { text: "Hybrid Retrieval", delay: "1.2s" },
  { text: "Cross-Encoder Rerank", delay: "1.8s" },
  { text: "Ollama LLM", delay: "2.4s" },
];

function FloatingChips() {
  return (
    <div className="flex flex-wrap justify-center gap-2 mb-6 select-none pointer-events-none">
      {FLOAT_CHIPS.map((c) => (
        <span
          key={c.text}
          className="text-[11px] font-medium text-accent-light/60 border border-accent/15 bg-accent/5 px-2.5 py-1 rounded-full"
          style={{ animation: `float 4s ease-in-out infinite`, animationDelay: c.delay }}
        >
          {c.text}
        </span>
      ))}
    </div>
  );
}

function SoundWave() {
  return (
    <div className="flex items-end gap-0.5 h-4">
      {[1, 2, 3, 4, 3, 2, 1].map((h, i) => (
        <span
          key={i}
          className="w-0.5 rounded-full bg-red-400"
          style={{
            height: `${h * 3 + 3}px`,
            animation: `soundBar 0.8s ease-in-out infinite alternate`,
            animationDelay: `${i * 0.1}s`,
          }}
        />
      ))}
    </div>
  );
}

export default function RagDemoPage() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [statusError, setStatusError] = useState("");
  const [topKTickets, setTopKTickets] = useState(3);
  const [topKKb, setTopKKb] = useState(3);
  const [useReranking, setUseReranking] = useState(true);
  const [description, setDescription] = useState("");
  const [running, setRunning] = useState<PipelineStep>(false);
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [error, setError] = useState("");
  const [stepLog, setStepLog] = useState<string[]>([]);
  const [interim, setInterim] = useState("");
  const [micError, setMicError] = useState("");
  const committedRef = useRef("");

  const handleFinal = useCallback((text: string) => {
    const appended = (committedRef.current + (committedRef.current ? " " : "") + text.trim()).trimStart();
    committedRef.current = appended;
    setDescription(appended);
    setInterim("");
  }, []);

  const handleInterim = useCallback((text: string) => {
    setInterim(text);
  }, []);

  const handleMicError = useCallback((msg: string) => {
    setMicError(msg);
    setTimeout(() => setMicError(""), 5000);
  }, []);

  const { listening, supported, start, stop } = useSpeechRecognition({
    onFinal: handleFinal,
    onInterim: handleInterim,
    onError: handleMicError,
  });

  const handleDescChange = (val: string) => {
    committedRef.current = val;
    setDescription(val);
    setInterim("");
  };

  function toggleMic() {
    if (listening) {
      stop();
      setInterim("");
    } else {
      setMicError("");
      start();
    }
  }

  useEffect(() => {
    getStatus().then(setStatus).catch((e) => setStatusError((e as Error).message));
  }, []);

  async function handleSubmit() {
    if (listening) stop();
    const text = description.trim();
    if (!text) {
      setError("Please enter a description.");
      return;
    }
    setError("");
    setResult(null);
    setStepLog([]);
    setRunning("tickets");

    const steps: PipelineStep[] = ["tickets", "kb", "llm"];
    let idx = 0;
    const timer = setInterval(() => {
      idx++;
      if (idx < steps.length) setRunning(steps[idx]);
    }, 1800);

    try {
      const data = await analyze(text, topKTickets, topKKb, useReranking);
      clearInterval(timer);
      setStepLog([
        `Found ${data.counts.tickets} similar tickets (${data.timings.tickets_s}s)`,
        `Found ${data.counts.kb} KB articles (${data.timings.kb_s}s)`,
        `LLM generated response (${data.timings.llm_s}s)`,
        `Total: ${data.timings.total_s}s`,
      ]);
      setResult(data);
    } catch (e: unknown) {
      clearInterval(timer);
      setError(e instanceof Error ? e.message : "Analysis failed.");
    } finally {
      setRunning(false);
    }
  }

  function handleReset() {
    if (listening) stop();
    setDescription("");
    committedRef.current = "";
    setInterim("");
    setResult(null);
    setError("");
    setStepLog([]);
    setRunning(false);
  }

  const canSubmit = !!description.trim() && !running;
  const displayValue = description + (interim ? (description ? " " : "") + interim : "");

  return (
    <>
      <NeuralBg />
      <CursorGlow />
      <div
        className="relative flex flex-col min-h-screen bg-surface-0/90 text-gray-200 z-10"
        style={{ cursor: "none" }}
      >
        <p className="px-4 pt-3 text-xs text-gray-500">
          <Link to="/" className="text-accent-light hover:underline" style={{ cursor: "pointer" }}>
            ← Home
          </Link>
        </p>
        <TopBar
          status={status}
          statusError={statusError}
          topKTickets={topKTickets}
          topKKb={topKKb}
          useReranking={useReranking}
          onTopKTickets={setTopKTickets}
          onTopKKb={setTopKKb}
          onUseReranking={setUseReranking}
          onReset={handleReset}
        />
        <main className="flex-1 overflow-y-auto">
          <div className="max-w-3xl mx-auto px-6 py-8 space-y-6">
            <div className="text-center pt-2 pb-4">
              <FloatingChips />
              <h2 className="text-3xl font-bold text-gray-100 tracking-tight mb-2">
                Submit a{" "}
                <span className="bg-gradient-to-r from-accent-light via-purple-400 to-accent bg-clip-text text-transparent">
                  New Ticket
                </span>
              </h2>
              <p className="text-sm text-gray-500 max-w-lg mx-auto">
                Describe your IT issue — the RAG pipeline retrieves relevant context from
                500 past tickets and 7 KB articles to generate a resolution blueprint.
              </p>
            </div>
            <div
              className="relative bg-surface-2/80 backdrop-blur-sm border border-border rounded-2xl p-6
              shadow-[0_4px_32px_rgba(0,0,0,0.4)] hover:border-accent/30 transition-colors duration-300 group"
            >
              <div className="absolute top-0 right-0 w-24 h-24 rounded-2xl overflow-hidden pointer-events-none">
                <div className="absolute top-0 right-0 w-16 h-16 bg-gradient-to-bl from-accent/10 to-transparent rounded-tl-full" />
              </div>
              <div className="flex items-center justify-between mb-3">
                <label className="text-xs font-semibold text-gray-400 uppercase tracking-widest">
                  Describe your issue
                </label>
                {supported && (
                  <button
                    type="button"
                    onClick={toggleMic}
                    title={listening ? "Stop recording" : "Start voice input"}
                    style={{ cursor: "none" }}
                    className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-sm font-medium
                      transition-all duration-200
                      ${
                        listening
                          ? "bg-red-500/15 border-red-500/40 text-red-400 shadow-[0_0_12px_rgba(239,68,68,0.25)]"
                          : "bg-surface-3 border-border text-gray-400 hover:bg-surface-4 hover:text-gray-200 hover:border-accent/30"
                      }`}
                  >
                    {listening ? (
                      <>
                        <MicOff size={14} /> <SoundWave /> Stop
                      </>
                    ) : (
                      <>
                        <Mic size={14} /> Voice Input
                      </>
                    )}
                  </button>
                )}
              </div>
              <div className="relative">
                <textarea
                  className={`w-full bg-surface-1 border rounded-xl px-4 py-3.5 text-sm
                    text-gray-200 placeholder-gray-600 focus:ring-2 focus:ring-accent/10
                    transition-all duration-200 resize-none leading-relaxed group-hover:border-surface-4
                    ${
                      listening
                        ? "border-red-500/40 focus:border-red-500/60"
                        : "border-border focus:border-accent/60"
                    }`}
                  rows={5}
                  placeholder={listening ? "Listening… speak now" : "e.g. My VPN disconnects every 10 minutes…"}
                  value={displayValue}
                  onChange={(e) => handleDescChange(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) handleSubmit();
                  }}
                  style={{ cursor: "text" }}
                />
                {listening && (
                  <div
                    className="absolute top-3 right-3 flex items-center gap-1.5 bg-red-500/20
                    border border-red-500/40 text-red-400 text-xs font-semibold px-2 py-1 rounded-full
                    shadow-[0_0_8px_rgba(239,68,68,0.3)]"
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse" />
                    REC
                  </div>
                )}
                {listening && interim && (
                  <div className="absolute bottom-3 left-3 right-3 pointer-events-none">
                    <p className="text-xs text-gray-500 italic truncate">
                      Hearing: <span className="text-accent-light">{interim}</span>
                    </p>
                  </div>
                )}
              </div>
              {micError && (
                <p className="mt-2 text-xs text-red-400 flex items-center gap-1.5">
                  <span>⚠</span> {micError}
                </p>
              )}
              <div className="flex items-center justify-between mt-4 gap-3">
                <p className="text-xs text-gray-700">
                  {listening
                    ? "Speaking… click Stop or press Ctrl+Enter to analyze"
                    : "Type or use Voice Input · Ctrl+Enter to submit"}
                </p>
                <button
                  type="button"
                  onClick={handleSubmit}
                  disabled={!canSubmit}
                  style={{ cursor: canSubmit ? "none" : "not-allowed" }}
                  className="flex items-center gap-2.5 px-6 py-2.5 rounded-xl text-sm font-bold
                    text-white transition-all duration-200 select-none
                    bg-gradient-to-r from-accent to-purple-700
                    hover:from-accent-hover hover:to-purple-800
                    disabled:opacity-40 disabled:cursor-not-allowed
                    shadow-lg shadow-accent/25 hover:shadow-accent/40
                    hover:scale-[1.02] active:scale-[0.98]"
                >
                  {running ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
                  {running ? "Analyzing…" : "Submit & Analyze"}
                </button>
              </div>
            </div>
            {error && (
              <div
                className="flex items-start gap-3 bg-yellow-500/8 border border-yellow-500/20
                text-yellow-300 text-sm rounded-xl px-4 py-3.5"
              >
                <span className="text-base leading-none mt-0.5">⚠️</span>
                <span>{error}</span>
              </div>
            )}
            {(running || stepLog.length > 0) && (
              <div
                className="bg-surface-2/80 backdrop-blur-sm border border-border rounded-2xl overflow-hidden
                shadow-[0_4px_24px_rgba(0,0,0,0.3)]"
              >
                <div
                  className="flex items-center gap-2.5 px-5 py-3.5 border-b border-border
                  bg-gradient-to-r from-surface-3 to-surface-2"
                >
                  {running ? (
                    <Loader2 size={13} className="animate-spin text-accent-light" />
                  ) : (
                    <span className="w-2.5 h-2.5 rounded-full bg-green-400 shadow-[0_0_8px_rgba(74,222,128,0.7)]" />
                  )}
                  <span className="text-sm font-semibold text-gray-200">
                    {running ? "Running RAG pipeline…" : "Pipeline complete"}
                  </span>
                  {!running && stepLog[3] && (
                    <span className="ml-auto text-xs text-green-400 font-semibold">✓ {stepLog[3]}</span>
                  )}
                </div>
                <div className="px-5 py-4 space-y-3">
                  {running && (
                    <div className="flex flex-wrap gap-2">
                      {STEP_META.map(({ key, label }) => {
                        const runIdx = STEP_META.findIndex((s) => s.key === running);
                        const stepIdx = STEP_META.findIndex((s) => s.key === key);
                        const done = stepIdx < runIdx;
                        const active = key === running;
                        return (
                          <span
                            key={key}
                            className={`flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-full border transition-all
                              ${
                                active
                                  ? "bg-accent/20 text-accent-light border-accent/40 shadow-[0_0_12px_rgba(124,58,237,0.25)]"
                                  : done
                                    ? "bg-green-500/12 text-green-400 border-green-500/25"
                                    : "bg-surface-3 text-gray-600 border-border"
                              }`}
                          >
                            {done ? (
                              "✓"
                            ) : active ? (
                              <Loader2 size={10} className="animate-spin" />
                            ) : (
                              <span className="w-1.5 h-1.5 rounded-full bg-gray-700" />
                            )}
                            {label}
                          </span>
                        );
                      })}
                    </div>
                  )}
                  {stepLog.slice(0, 3).map((msg, i) => (
                    <div key={i} className="flex items-center gap-2.5 text-xs text-gray-400">
                      <Sparkles size={10} className="text-accent-light flex-shrink-0" />
                      {msg}
                    </div>
                  ))}
                </div>
              </div>
            )}
            {result && <ResultPanel result={result} />}
          </div>
        </main>
      </div>
    </>
  );
}
