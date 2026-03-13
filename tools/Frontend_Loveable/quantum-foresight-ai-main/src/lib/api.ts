/**
 * api.ts — Lachesis API client
 * All calls go to the FastAPI server at VITE_API_URL (default: http://localhost:8000)
 */

const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `API error ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, { method: "POST", body: JSON.stringify(body) });
}

export function get<T>(path: string): Promise<T> {
  return request<T>(path, { method: "GET" });
}

// ─── Auth (admin signup via backend) ─────────────────────────────────────────
export const apiSignUp = (email: string, password: string, display_name: string) =>
  post<{ success: boolean }>("/api/auth/signup", { email, password, display_name });

// ─── Health ──────────────────────────────────────────────────────────────────
export interface HealthResponse {
  status: string;
  version: string;
  capabilities: Record<string, boolean>;
}
export const apiHealth = () => get<HealthResponse>("/api/health");

// ─── Web Search (SerpAPI) ─────────────────────────────────────────────────────
export interface SearchResult {
  title: string;
  snippet: string;
  link: string;
}
export interface SearchResponse {
  results: SearchResult[];
  query: string;
}
export const apiWebSearch = (query: string, serpapi_key: string, num_results = 8) =>
  post<SearchResponse>("/api/search", { query, serpapi_key, num_results });

// ─── Quantum Simulation ───────────────────────────────────────────────────────
export interface GateStep {
  q0?: string;
  q0_angle?: number;
  q1?: string;
  q1_angle?: number;
  q2?: string;
  q2_angle?: number;
  q3?: string;
  q3_angle?: number;
  cnot_01?: boolean;
  cnot_12?: boolean;
  cnot_23?: boolean;
}

export interface NoiseParams {
  enable_depolarizing?: boolean;
  depolarizing_prob?: number;
  enable_amplitude_damping?: boolean;
  amplitude_damping_prob?: number;
  enable_phase_damping?: boolean;
  phase_damping_prob?: number;
  enable_cnot_noise?: boolean;
  cnot_noise_prob?: number;
}

export interface QuantumSimulateRequest {
  num_qubits?: number;
  shots?: number;
  seed?: number;
  step0?: GateStep;
  step1?: GateStep;
  step2?: GateStep;
  noise?: NoiseParams;
}

export interface QuantumSimulateResponse {
  statevector_real: number[];
  statevector_imag: number[];
  probabilities: number[];
  counts: Record<string, number>;
  counts_normalised: Record<string, number>;
  fidelity: number;
  circuit_lines: string[];
  num_qubits: number;
}

export const apiQuantumSimulate = (req: QuantumSimulateRequest) =>
  post<QuantumSimulateResponse>("/api/quantum/simulate", req);

// ─── Advanced Quantum ─────────────────────────────────────────────────────────
export interface TomographyResponse {
  bloch_x: number;
  bloch_y: number;
  bloch_z: number;
  purity: number;
}
export const apiQuantumTomography = (gate: string, angle: number, shots = 4096, seed?: number) =>
  post<TomographyResponse>("/api/quantum/advanced/tomography", { gate, angle, shots, seed });

export interface BenchmarkingResponse {
  lengths: number[];
  survival: number[];
  fit: { A: number; p: number; B: number };
  EPG: number;
}
export const apiQuantumBenchmarking = (lengths: number[], nseeds = 8, shots = 2048, seed?: number) =>
  post<BenchmarkingResponse>("/api/quantum/advanced/benchmarking", { lengths, nseeds, shots, seed });

export interface CalibrationPosterior {
  alpha: number;
  beta: number;
  mean: number;
  ci_low: number;
  ci_high: number;
  gate_label: string;
}
export interface CalibrateResponse {
  posteriors: Record<string, CalibrationPosterior>;
  shots: number;
}
export const apiQuantumCalibrate = (shots = 4096, seed?: number) =>
  post<CalibrateResponse>("/api/quantum/advanced/calibrate", { shots, seed });

export interface FidelityResponse {
  fidelity: number;
  per_basis: number[];
  gate: string;
}
export const apiQuantumFidelity = (gate: string, angle: number, shots = 4096, seed?: number) =>
  post<FidelityResponse>("/api/quantum/advanced/fidelity", { gate, angle, shots, seed });

// ─── Financial Analytics ──────────────────────────────────────────────────────
export interface FinancialAnalyzeRequest {
  tickers: string[];
  lookback_days?: number;
  confidence?: number;
  simulations?: number;
  demo_mode?: boolean;
  /** Optional sentiment multiplier from apiSentimentAnalyze → stresses VaR/CVaR */
  sentiment_multiplier?: number | null;
  /** Use Quantum Amplitude Estimation instead of classical MC (requires qiskit-finance) */
  use_qae?: boolean;
}

export interface FinancialAnalyzeResponse {
  tickers: string[];
  dates: string[];
  prices: Record<string, number[]>;
  returns: Record<string, number[]>;
  portfolio_returns: number[];
  var_mc: number;
  cvar_mc: number;
  var_historical: number;
  cvar_historical: number;
  sharpe: number;
  sortino: number;
  max_drawdown: number;
  annualized_volatility: number;
  regime: string;
  data_source: string;
  sentiment_multiplier?: number | null;
  var_mc_stressed?: number | null;
  cvar_mc_stressed?: number | null;
  use_qae?: boolean;
  qae_active?: boolean;
  qae_available?: boolean;
  qae_tail_prob?: number | null;
}

export const apiFinancialAnalyze = (req: FinancialAnalyzeRequest) =>
  post<FinancialAnalyzeResponse>("/api/financial/analyze", req);

// ─── QTBN Forecast ────────────────────────────────────────────────────────────
export interface QTBNForecastRequest {
  prior_regime?: string;
  risk_on_prior?: number;
  drift_mu?: number;
  horizon_days?: number;
  steps?: number;
}

export interface QTBNForecastResponse {
  prior_regime: string;
  horizon_days: number;
  P_gain: number;
  P_flat: number;
  P_loss: number;
  P_severe_loss: number;
  regime_timeline: Array<Record<string, number>>;
  drift_path: number[];
  risk_on_path: number[];
}

export const apiQTBNForecast = (req: QTBNForecastRequest) =>
  post<QTBNForecastResponse>("/api/qtbn/forecast", req);

// ─── QAOA ─────────────────────────────────────────────────────────────────────
export interface QAOAPortfoliosResponse {
  portfolios: string[];
  toy?: Record<string, unknown>;
  benchmark?: Record<string, unknown>;
  note?: string;
}
export const apiQAOAPortfolios = () => get<QAOAPortfoliosResponse>("/api/qaoa/portfolios");

export interface QAOAOptimizeRequest {
  portfolio?: string;
  depth?: number;
  shots?: number;
  lam?: number;
  backend?: string;
  regime?: string | null;
}

export interface QAOAOptimizeResponse {
  bitstring: string;
  selected_assets: string[];
  expected_return: number;
  risk: number;
  objective: number;
  energy: number;
  backend: string;
  lam: number;
  narrative: string;
  assets: string[];
}
export const apiQAOAOptimize = (req: QAOAOptimizeRequest) =>
  post<QAOAOptimizeResponse>("/api/qaoa/optimize", req);

export interface QAOASweepPoint {
  lam: number;
  expected_return: number;
  risk: number;
  objective: number;
  selected_assets: string[];
  bitstring: string;
}
export interface QAOASweepResponse {
  sweep: QAOASweepPoint[];
  portfolio: string;
}
export const apiQAOASweep = (portfolio: string, lam_min: number, lam_max: number, n_points: number) =>
  post<QAOASweepResponse>("/api/qaoa/sweep", { portfolio, lam_min, lam_max, n_points });

export interface QAOAScenario {
  name: string;
  result: Record<string, unknown>;
  portfolio: string;
  notes?: string;
  timestamp?: string;
}
export const apiQAOAGetScenarios = () =>
  get<{ scenarios: QAOAScenario[] }>("/api/qaoa/scenarios");

export const apiQAOASaveScenario = (name: string, result: Record<string, unknown>, portfolio: string, notes = "") =>
  post<{ status: string; name: string }>("/api/qaoa/scenarios", { name, result, portfolio, notes });

export const apiQAOAGetLog = () => get<{ rows: Record<string, string>[] }>("/api/qaoa/log");

// ─── VQE Risk Gate ────────────────────────────────────────────────────────────
export interface VQERiskGateRequest {
  requested_notional_usd: number;
  price_usd?: number;
  vol_daily_pct?: number;
  leverage?: number;
  policy?: string;
}

export interface VQERiskGateResponse {
  timestamp: string;
  policy: string;
  requested_notional_usd: number;
  final_notional_usd: number;
  est_var_usd: number;
  est_cvar_usd: number;
  leverage_used: number;
  status: "APPROVED" | "PARTIAL" | "BLOCKED";
  reasons: string[];
  limits: Record<string, number>;
}
export const apiVQERiskGate = (req: VQERiskGateRequest) =>
  post<VQERiskGateResponse>("/api/vqe/risk-gate", req);

export const apiVQEAudit = (limit = 20) =>
  get<{ records: VQERiskGateResponse[]; total: number }>(`/api/vqe/audit?limit=${limit}`);

// ─── VQE Solve (Advanced Metrics) ─────────────────────────────────────────────
export interface VQESolveRequest {
  problem?: string;
  ansatz_name?: string;
  optimizer_name?: string;
  num_qubits?: number;
  reps?: number;
  maxiter?: number;
  seed?: number | null;
  pauli_text?: string;
  maxcut_edges_text?: string;
  ising_h_text?: string;
  ising_J_text?: string;
  backend_choice?: string;
}
export interface VQESolveResponse {
  converged: boolean;
  used_fallback: boolean;
  energy: number | null;
  risk_multiplier: number;
  estimator: string;
  history: { t: number; energy?: number; eval_count?: number }[];
  num_pauli_terms: number;
  problem_type: string;
  ansatz_desc: string;
  optimizer_desc: string;
  num_qubits: number;
}
export const apiVQESolve = (req: VQESolveRequest) =>
  post<VQESolveResponse>("/api/vqe/solve", req);

// ─── Foresight ────────────────────────────────────────────────────────────────
export interface ForesightCell {
  pdep: number;
  pamp: number;
  kl_divergence: number;
  counts: Record<string, number>;
}
export interface ForesightSweepResponse {
  pdep_values: number[];
  pamp_values: number[];
  grid: ForesightCell[][];
}
export const apiForesightSweep = (body: {
  shots: number;
  seeds: number[];
  pdep_values: number[];
  pamp_values: number[];
  circuit: QuantumSimulateRequest;
}) => post<ForesightSweepResponse>("/api/foresight/sweep", body);

export const apiForesightGetScenarios = () =>
  get<{ scenarios: Record<string, unknown> }>("/api/foresight/scenarios");

export const apiForesightSaveScenario = (name: string, data: Record<string, unknown>) =>
  post<{ status: string; name: string }>("/api/foresight/scenarios", { name, data });

// ─── Sentiment ────────────────────────────────────────────────────────────────
export interface SentimentItem {
  ticker: string;
  title: string;
  score: number;
  published: string;
  link: string;
}
export interface SentimentResponse {
  tickers: string[];
  items: SentimentItem[];
  headlines: string[];
  avg_score: number;
  multiplier: number;
  total_items: number;
  provider?: string;
}

export interface SentimentAnalyzeRequest {
  tickers: string[];
  keywords?: string[];
  max_items?: number;
  provider?: "google_rss" | "perplexity";
  perplexity_api_key?: string;
  perplexity_model?: string;
}

export const apiSentimentAnalyze = (req: SentimentAnalyzeRequest) =>
  post<SentimentResponse>("/api/sentiment/analyze", req);

// ─── Reduced States ───────────────────────────────────────────────────────────
export interface ReducedState {
  qubit: number;
  bloch_x: number; bloch_y: number; bloch_z: number;
  purity: number;
  rho_real: number[][]; rho_imag: number[][];
}
export interface ReducedStatesResponse {
  num_qubits: number;
  reduced_states: ReducedState[];
  noise_applied?: boolean;
}
export const apiQuantumReducedStates = (req: QuantumSimulateRequest) =>
  post<ReducedStatesResponse>("/api/quantum/reduced-states", req);

// ─── Measurement (ideal vs noisy) ────────────────────────────────────────────
export interface MeasurementResponse {
  ideal_counts: Record<string, number>;
  noisy_counts: Record<string, number>;
  ideal_probs: Record<string, number>;
  noisy_probs: Record<string, number>;
  tv_distance: number;
  all_states: string[];
  num_qubits: number;
}
export const apiQuantumMeasurement = (req: QuantumSimulateRequest) =>
  post<MeasurementResponse>("/api/quantum/measurement", req);

// ─── Presets ─────────────────────────────────────────────────────────────────
export interface PresetMeta { key: string; label: string; }
export const apiGetPresets = () => get<{ presets: PresetMeta[] }>("/api/quantum/presets");
export const apiGetPreset  = (key: string) => get<Record<string, unknown>>(`/api/quantum/presets/${key}`);

// ─── Financial: Insider ───────────────────────────────────────────────────────
export interface InsiderRequest {
  tickers: string[];
  lookback_days?: number;
  portfolio_value?: number;
  confidence?: number;
  simulations?: number;
  demo_mode?: boolean;
}
export interface PerAssetStat {
  ticker: string;
  ann_return_pct: number;
  ann_vol_pct: number;
  sharpe: number;
  max_drawdown_pct: number;
  last_price: number;
}
export interface InsiderResponse {
  tickers: string[];
  data_source: string;
  portfolio_value: number;
  var_1d_usd: number;
  cvar_1d_usd: number;
  regime: string;
  current_vol_ann_pct: number;
  per_asset: PerAssetStat[];
  positions: Array<{ ticker: string; weight_pct: number; value_usd: number }>;
}
export const apiFinancialInsider = (req: InsiderRequest) =>
  post<InsiderResponse>("/api/financial/insider", req);

// ─── Lachesis Guide ───────────────────────────────────────────────────────────
export interface LachesisGuideRequest {
  question: string;
  tickers?: string[];
  regime?: string;
  var_usd?: number | null;
  cvar_usd?: number | null;
  portfolio_value?: number;
  openai_api_key?: string | null;
  language?: string;
}
export interface LachesisGuideResponse {
  narrative: string;
  context: string;
  question: string;
}
export const apiLachesisGuide = (req: LachesisGuideRequest) =>
  post<LachesisGuideResponse>("/api/financial/lachesis-guide", req);

// ─── Prompt Studio ────────────────────────────────────────────────────────────
export interface PromptTemplate { key: string; template: string; }
export interface PromptGenerateResponse {
  prompt: string;
  result: string;
  template: string;
  tokens_requested: number;
}
export const apiPromptTemplates = () =>
  get<{ templates: PromptTemplate[] }>("/api/prompt-studio/templates");
export const apiPromptGenerate = (
  template: string,
  variables: Record<string, unknown>,
  customPrompt?: string,
  openaiApiKey?: string,
  maxTokens = 500,
  language = "English",
) =>
  post<PromptGenerateResponse>("/api/prompt-studio/generate", {
    template, variables, custom_prompt: customPrompt ?? null,
    openai_api_key: openaiApiKey ?? null, max_tokens: maxTokens, language,
  });

// ─── Admin key validation ─────────────────────────────────────────────────────
export const apiAdminValidateKey = (service: string, api_key: string) =>
  post<{ service: string; valid: boolean; hint: string }>("/api/admin/validate-key", { service, api_key });

// ─── SEC EDGAR ────────────────────────────────────────────────────────────────
export interface EdgarCIKResponse {
  ticker: string;
  cik: string;
  company_name: string;
}

export interface EdgarFiling {
  accession_number: string;
  filing_date: string;
  form: string;
  primary_document: string;
  description: string;
  filing_url: string;
}

export interface EdgarLoadResponse {
  ticker: string;
  cik: string;
  company_name: string;
  filings: EdgarFiling[];
  total_found: number;
}

/** Single-call endpoint: provide ticker OR manual CIK — mirrors Streamlit "Load filings" */
export const apiEdgarLoadFilings = (
  ticker: string,
  cik: string,
  forms: string[],
  user_agent: string,
  max_results = 50,
) =>
  post<EdgarLoadResponse>("/api/insider/load-filings", {
    ticker, cik, forms, user_agent, max_results,
  });

// Legacy aliases kept for any remaining callers
export type EdgarFilingsResponse = EdgarLoadResponse;
export const apiEdgarLookupCIK = (ticker: string, user_agent: string) =>
  post<EdgarCIKResponse>("/api/insider/lookup-cik", { ticker, user_agent });
export const apiEdgarFilings = (cik: string, forms: string[], user_agent: string, max_results = 50) =>
  post<EdgarLoadResponse>("/api/insider/filings", { cik, forms, user_agent, max_results });
