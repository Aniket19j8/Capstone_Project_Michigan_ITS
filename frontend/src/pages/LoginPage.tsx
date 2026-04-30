import { useEffect, useState } from "react";
import { useNavigate, useSearchParams, Link } from "react-router-dom";
import { Headphones, Loader2, LogIn, Lock, Mail, UserRound } from "lucide-react";
import { login } from "../api";
import NeuralBg from "../components/NeuralBg";

export default function LoginPage() {
  const [sp] = useSearchParams();
  const role = sp.get("role") === "admin" ? "admin" : "user";
  const [email, setEmail] = useState(role === "admin" ? "admin@demo.com" : "user@demo.com");
  const [password, setPassword] = useState("demo");
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);
  const nav = useNavigate();

  useEffect(() => {
    setEmail(role === "admin" ? "admin@demo.com" : "user@demo.com");
  }, [role]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setErr("");
    setLoading(true);
    try {
      const r = await login(email.trim().toLowerCase(), password);
      localStorage.setItem("its_token", r.access_token);
      localStorage.setItem("its_user", JSON.stringify(r.user));
      nav(r.user.role === "admin" ? "/admin" : "/support", { replace: true });
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Login failed");
    } finally {
      setLoading(false);
    }
  }

  const isAdmin = role === "admin";

  return (
    <>
      <NeuralBg />
      <div
        className="relative z-10 min-h-screen flex items-center justify-center px-4 bg-surface-0/85"
        style={{ cursor: "auto" }}
      >
        <form
          onSubmit={onSubmit}
          className="w-full max-w-md bg-surface-2/90 backdrop-blur border border-border rounded-2xl p-8 shadow-2xl shadow-black/40 relative overflow-hidden"
        >
          <div
            className={`absolute -top-24 -right-20 w-56 h-56 rounded-full blur-3xl ${
              isAdmin ? "bg-cyan-500/20" : "bg-accent/25"
            }`}
          />
          <div className="relative flex items-center gap-3 mb-6">
            <div
              className={`w-11 h-11 rounded-xl flex items-center justify-center shadow-lg ${
                isAdmin
                  ? "bg-gradient-to-br from-cyan-500 to-blue-700 shadow-cyan-500/30"
                  : "bg-gradient-to-br from-accent to-purple-700 shadow-accent/30"
              }`}
            >
              {isAdmin ? (
                <Headphones size={20} className="text-white" />
              ) : (
                <UserRound size={20} className="text-white" />
              )}
            </div>
            <div>
              <h2 className="text-lg font-bold text-gray-100 leading-tight">
                {isAdmin ? "Admin sign in" : "User sign in"}
              </h2>
              <p className="text-xs text-gray-500">
                Demo build · password <code className="text-gray-400">demo</code>
              </p>
            </div>
          </div>

          <label className="relative block text-[11px] uppercase tracking-widest text-gray-500 mb-1.5">
            Email
          </label>
          <div className="relative mb-4">
            <Mail
              size={14}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500 pointer-events-none"
            />
            <input
              className="w-full bg-surface-1 border border-border rounded-lg pl-9 pr-3 py-2.5 text-sm text-gray-200 focus:border-accent/50 transition-colors"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="email"
            />
          </div>

          <label className="relative block text-[11px] uppercase tracking-widest text-gray-500 mb-1.5">
            Password
          </label>
          <div className="relative mb-5">
            <Lock
              size={14}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500 pointer-events-none"
            />
            <input
              type="password"
              className="w-full bg-surface-1 border border-border rounded-lg pl-9 pr-3 py-2.5 text-sm text-gray-200 focus:border-accent/50 transition-colors"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>

          {err && (
            <p className="relative text-red-400 text-xs mb-3 bg-red-500/10 border border-red-500/20 px-3 py-2 rounded-lg">
              {err}
            </p>
          )}

          <button
            type="submit"
            disabled={loading}
            className={`relative w-full py-2.5 rounded-lg text-white text-sm font-semibold flex items-center justify-center gap-2 transition-all active:scale-[0.99] ${
              isAdmin
                ? "bg-gradient-to-r from-cyan-500 to-blue-700 hover:from-cyan-400 hover:to-blue-600 shadow-lg shadow-cyan-500/25"
                : "bg-gradient-to-r from-accent to-purple-700 hover:from-accent-hover hover:to-purple-800 shadow-lg shadow-accent/25"
            } disabled:opacity-50 disabled:saturate-50`}
          >
            {loading ? (
              <Loader2 className="animate-spin" size={16} />
            ) : (
              <>
                <LogIn size={14} /> Continue
              </>
            )}
          </button>

          <p className="relative mt-5 text-center text-xs text-gray-500">
            <Link to="/" className="text-accent-light hover:underline">
              ← Back to home
            </Link>
          </p>
        </form>
      </div>
    </>
  );
}
