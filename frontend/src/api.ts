import type { SystemStatus, AnalyzeResult } from "./types";

const BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

function authHeader(): Record<string, string> {
  const t = localStorage.getItem("its_token");
  if (!t) return {};
  return { Authorization: `Bearer ${t}` };
}

export async function getStatus(): Promise<SystemStatus> {
  const r = await fetch(`${BASE}/api/status`);
  if (!r.ok) throw new Error("Backend unreachable");
  return r.json();
}

export async function analyze(
  description: string,
  topKTickets: number,
  topKKb: number,
  useReranking: boolean
): Promise<AnalyzeResult> {
  const r = await fetch(`${BASE}/api/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      description,
      top_k_tickets: topKTickets,
      top_k_kb: topKKb,
      use_reranking: useReranking,
    }),
  });
  if (!r.ok) {
    const err = (await r.json().catch(() => ({}))) as { detail?: string };
    throw new Error(err.detail ?? "Analysis failed");
  }
  return r.json();
}

export type LoginRes = {
  access_token: string;
  token_type: string;
  user: { id: number; email: string; name: string; role: string; display_email: string };
};

export async function login(email: string, password: string): Promise<LoginRes> {
  const r = await fetch(`${BASE}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!r.ok) {
    const e = (await r.json().catch(() => ({}))) as { detail?: string | unknown };
    const m = typeof e.detail === "string" ? e.detail : "Login failed";
    throw new Error(m);
  }
  return r.json();
}

export async function createSession(): Promise<{ session_id: string; user_id: number }> {
  const r = await fetch(`${BASE}/api/sessions`, {
    method: "POST",
    headers: { ...authHeader() },
  });
  if (!r.ok) throw new Error("Could not start session");
  return r.json();
}

export type ChatMessage = { id: number; sender: string; content: string; created_at: string | null };

export type SessionState = {
  session_id: string;
  user_id: number;
  state: string;
  ai_rounds: number;
  needs_escalation_choice: boolean;
  departments: string[];
  messages: ChatMessage[];
  ticket: {
    id: number;
    public_id: string;
    user_id: number;
    issue_summary: string;
    department: string;
    department_confidence?: number | null;
    routing_reason?: string | null;
    triage_scores?: Record<string, number>;
    status: string;
  } | null;
};

export async function getSession(sessionId: string): Promise<SessionState> {
  const r = await fetch(`${BASE}/api/sessions/${encodeURIComponent(sessionId)}`, {
    headers: { ...authHeader() },
  });
  if (!r.ok) throw new Error("Session not found");
  return r.json();
}

export type PostMessageRes = {
  session_id: string;
  messages: ChatMessage[];
  last_model?: {
    confidence: number;
    ok: boolean;
    recommended_steps?: string[];
    escalation_recommended?: boolean;
    retrieval_score?: number | null;
  } | null;
  suggest_specialist?: boolean;
  needs_escalation_choice?: boolean;
};

export async function postMessage(sessionId: string, text: string): Promise<PostMessageRes> {
  const r = await fetch(`${BASE}/api/sessions/${encodeURIComponent(sessionId)}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeader() },
    body: JSON.stringify({ text }),
  });
  if (!r.ok) {
    const e = (await r.json().catch(() => ({}))) as { detail?: string };
    throw new Error(typeof e.detail === "string" ? e.detail : "Message failed");
  }
  return r.json();
}

export async function postAction(
  sessionId: string,
  action: "retry" | "escalate"
): Promise<{
  ok: boolean;
  reset?: boolean;
  ticket?: {
    id: number;
    public_id: string;
    user_id: number;
    issue_summary: string;
    department: string;
    department_confidence?: number | null;
    routing_reason?: string | null;
    triage_scores?: Record<string, number>;
    status: string;
    session_id: string;
  };
}> {
  const r = await fetch(`${BASE}/api/sessions/${encodeURIComponent(sessionId)}/action`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeader() },
    body: JSON.stringify({ action }),
  });
  if (!r.ok) {
    const e = (await r.json().catch(() => ({}))) as { detail?: string };
    throw new Error(typeof e.detail === "string" ? e.detail : "Action failed");
  }
  return r.json();
}

export type AdminTicketRow = {
  id: number;
  public_id: string;
  user_id: number;
  user_email: string;
  department: string;
  department_confidence?: number | null;
  routing_reason?: string | null;
  triage_scores?: Record<string, number>;
  status: string;
  created_at: string | null;
  issue_excerpt: string;
};

export async function listAdminTickets(): Promise<{ tickets: AdminTicketRow[] }> {
  const r = await fetch(`${BASE}/api/admin/tickets`, { headers: { ...authHeader() } });
  if (!r.ok) throw new Error("Failed to list tickets");
  return r.json();
}

export type AdminDetail = {
  ticket: {
    id: number;
    public_id: string;
    user_id: number;
    user_email: string;
    user_name: string;
    department: string;
    department_confidence?: number | null;
    routing_reason?: string | null;
    triage_scores?: Record<string, number>;
    status: string;
    issue_summary: string;
    session_id: string;
    created_at: string | null;
  };
  messages: ChatMessage[];
  fake_outbound_email: { to: string; subject: string; body: string };
};

export async function getAdminTicket(ticketId: number): Promise<AdminDetail> {
  const r = await fetch(`${BASE}/api/admin/tickets/${ticketId}`, { headers: { ...authHeader() } });
  if (!r.ok) throw new Error("Ticket not found");
  return r.json();
}

export type Insights = {
  ticket_id: number;
  public_id: string;
  user_id: number;
  resolution: string;
  similar_tickets: unknown[];
  kb_articles: unknown[];
  timings: Record<string, number>;
};

export async function getCachedAdminInsights(ticketId: number): Promise<Insights | null> {
  const r = await fetch(`${BASE}/api/admin/tickets/${ticketId}/insights`, {
    headers: { ...authHeader() },
  });
  if (r.status === 404) return null;
  if (!r.ok) throw new Error("Insights failed");
  return r.json();
}

export async function generateAdminInsights(ticketId: number): Promise<Insights> {
  const r = await fetch(`${BASE}/api/admin/tickets/${ticketId}/insights/generate`, {
    method: "POST",
    headers: { ...authHeader() },
  });
  if (!r.ok) throw new Error("Generate insights failed");
  return r.json();
}

export type CreateTicketRes = {
  ticket: {
    id: number;
    public_id: string;
    department: string;
    department_confidence?: number | null;
    routing_reason?: string | null;
    triage_scores?: Record<string, number>;
    status: string;
    resolution: string;
  };
  insights: Insights;
};

export async function adminCreateTicket(
  issueDescription: string,
  department: string,
  resolution: string
): Promise<CreateTicketRes> {
  const r = await fetch(`${BASE}/api/admin/create-ticket`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeader() },
    body: JSON.stringify({
      issue_description: issueDescription,
      department,
      resolution,
    }),
  });
  if (!r.ok) {
    const e = (await r.json().catch(() => ({}))) as { detail?: string };
    throw new Error(typeof e.detail === "string" ? e.detail : "Create ticket failed");
  }
  return r.json();
}

export async function refreshAdminInsights(ticketId: number): Promise<Insights> {
  const r = await fetch(`${BASE}/api/admin/tickets/${ticketId}/insights/refresh`, {
    method: "POST",
    headers: { ...authHeader() },
  });
  if (!r.ok) throw new Error("Refresh failed");
  return r.json();
}
