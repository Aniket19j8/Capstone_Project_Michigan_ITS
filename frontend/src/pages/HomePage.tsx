import { Link } from "react-router-dom";
import { ArrowRight, Bot, Database, Headphones, Sparkles, Ticket, UserRound } from "lucide-react";
import NeuralBg from "../components/NeuralBg";

function FeaturePill({
  icon,
  label,
}: {
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-medium border border-border bg-surface-2/80 text-gray-400">
      <span className="text-accent-light">{icon}</span>
      {label}
    </span>
  );
}

function RouteCard({
  to,
  title,
  subtitle,
  icon,
  primary,
}: {
  to: string;
  title: string;
  subtitle: string;
  icon: React.ReactNode;
  primary?: boolean;
}) {
  return (
    <Link
      to={to}
      className={`group relative overflow-hidden rounded-2xl p-6 border backdrop-blur transition-all ${
        primary
          ? "border-accent/40 bg-gradient-to-br from-accent/15 via-surface-2/80 to-surface-2/80 shadow-lg shadow-accent/10 hover:shadow-accent/30"
          : "border-border bg-surface-2/80 hover:border-accent/30"
      }`}
    >
      <div
        className={`absolute -top-12 -right-12 w-40 h-40 rounded-full blur-3xl transition-opacity ${
          primary ? "bg-accent/25 opacity-70" : "bg-accent/10 opacity-50 group-hover:opacity-80"
        }`}
      />
      <div
        className={`relative inline-flex w-10 h-10 items-center justify-center rounded-xl mb-4 ${
          primary
            ? "bg-gradient-to-br from-accent to-purple-700 text-white shadow-md shadow-accent/40"
            : "bg-surface-3 border border-border text-accent-light"
        }`}
      >
        {icon}
      </div>
      <h3 className="relative text-base font-semibold text-gray-100">{title}</h3>
      <p className="relative mt-1 text-[13px] text-gray-400 leading-relaxed">{subtitle}</p>
      <div className="relative mt-4 inline-flex items-center gap-1 text-xs font-medium text-accent-light">
        Launch <ArrowRight size={12} className="transition-transform group-hover:translate-x-0.5" />
      </div>
    </Link>
  );
}

export default function HomePage() {
  return (
    <>
      <NeuralBg />
      <div
        className="relative z-10 min-h-screen flex flex-col items-center px-6 py-14 md:py-20 text-gray-200 bg-surface-0/80"
        style={{ cursor: "auto" }}
      >
        <div className="inline-flex items-center gap-2 text-[11px] font-medium uppercase tracking-widest text-accent-light/80 border border-accent/20 bg-accent/5 px-3 py-1 rounded-full mb-6">
          <Sparkles size={12} /> Capstone Demo · Team Michigan
        </div>
        <h1 className="text-4xl md:text-5xl font-bold text-center tracking-tight mb-3">
          <span className="bg-gradient-to-r from-accent-light via-cyan-300 to-accent bg-clip-text text-transparent">
            Intelligent
          </span>{" "}
          <span className="text-gray-100">Ticketing System</span>
        </h1>
        <p className="text-sm md:text-base text-gray-400 max-w-xl text-center leading-relaxed">
          A chat-first support experience backed by two AI models, a RAG retrieval layer, and a
          specialist workbench — all running on your machine.
        </p>

        <div className="mt-6 flex flex-wrap justify-center gap-2">
          <FeaturePill icon={<Bot size={12} />} label="Model 1 · Resolution" />
          <FeaturePill icon={<Ticket size={12} />} label="Model 2 · Smart Triage" />
          <FeaturePill icon={<Database size={12} />} label="RAG + Knowledge Base" />
          <FeaturePill icon={<Headphones size={12} />} label="Live Voice Input" />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-12 max-w-4xl w-full">
          <RouteCard
            to="/login?role=user"
            title="User · Support Chat"
            subtitle="Describe an issue, talk or type, let the AI try twice, escalate if needed."
            icon={<UserRound size={18} />}
            primary
          />
          <RouteCard
            to="/login?role=admin"
            title="Admin · Workbench"
            subtitle="Triage tickets with similar-case retrieval, LLM recommendations and draft emails."
            icon={<Headphones size={18} />}
          />
          <RouteCard
            to="/rag"
            title="RAG Playground"
            subtitle="Inspect the retrieval + rerank pipeline directly over synthetic tickets."
            icon={<Sparkles size={18} />}
          />
        </div>

        <p className="mt-10 text-[11px] text-gray-600">
          Demo logins:&nbsp;
          <code className="text-gray-400">user@demo.com</code> ·{" "}
          <code className="text-gray-400">admin@demo.com</code> — password{" "}
          <code className="text-gray-400">demo</code>
        </p>
      </div>
    </>
  );
}
