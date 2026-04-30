export interface SystemStatus {
  retriever_ready: boolean;
  retriever_error: string | null;
  ticket_count: number | null;
  kb_count: number | null;
  llm_ready: boolean;
  llm_model: string;
}

export interface SimilarTicket {
  id: string;
  text: string;
  score: number;
  title: string;
  category: string;
  severity: string;
  status: string;
  component: string;
  ticket_id: string;
  source_file: string;
}

export interface KbArticle {
  id: string;
  text: string;
  score: number;
  title: string;
  source_file: string;
}

export interface AnalyzeResult {
  resolution: string;
  similar_tickets: SimilarTicket[];
  kb_articles: KbArticle[];
  timings: {
    tickets_s: number;
    kb_s: number;
    llm_s: number;
    total_s: number;
  };
  counts: {
    tickets: number;
    kb: number;
  };
}
