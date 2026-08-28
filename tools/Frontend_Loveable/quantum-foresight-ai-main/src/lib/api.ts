/**
 * api.ts — Lachesis API client
 * All calls go to the FastAPI server at VITE_API_URL (default: http://localhost:8000)
 */

import { supabase } from "@/integrations/supabase/client";

const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options?.headers as Record<string, string> | undefined) },
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

// ─── Auth-aware request helpers (attaches the caller's Supabase session token) ─
// Use these for any endpoint that requires Depends(_get_authenticated_user_id)
// on the backend — currently: Budgeting, Retirement, Credit Behavior Simulator.

async function authHeaders(): Promise<Record<string, string>> {
  const { data } = await supabase.auth.getSession();
  const token = data.session?.access_token;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function authGet<T>(path: string): Promise<T> {
  return request<T>(path, { method: "GET", headers: await authHeaders() });
}

export async function authPost<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, { method: "POST", headers: await authHeaders(), body: JSON.stringify(body) });
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

// ─── Lachesis AI chat + voice (server-side proxy — no client-supplied key) ────
export interface LachesisChatResponse {
  content: string;
  toolCall?: { id: string; type: string; function: { name: string; arguments: string } } | null;
}
export const apiLachesisChat = (messages: Array<{ role: string; content: unknown }>) =>
  post<LachesisChatResponse>("/api/lachesis/chat", { messages });

export async function apiLachesisSpeak(text: string): Promise<Blob> {
  const res = await fetch(`${BASE}/api/lachesis/speak`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(`Voice request failed: ${res.status}`);
  return res.blob();
}

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
  qasm_str?: string;
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

export interface QASMValidateResponse {
  valid: boolean;
  num_qubits: number;
  num_clbits: number;
  depth: number;
  num_gates: number;
  circuit_lines: string[];
  error: string | null;
}
export const apiQASMValidate = (qasm_str: string) =>
  post<QASMValidateResponse>("/api/quantum/qasm-validate", { qasm_str });

// ─── IBM Quantum ──────────────────────────────────────────────────────────────
export interface IBMBackend {
  name: string;
  num_qubits: number | null;
  operational: boolean;
  pending_jobs: number;
  simulator: boolean;
}
export interface IBMListBackendsResponse {
  backends: IBMBackend[];
  total: number;
}
export const apiIBMListBackends = (ibm_token: string) =>
  post<IBMListBackendsResponse>("/api/ibm/list-backends", { ibm_token });

export interface IBMRunCircuitResponse {
  backend: string;
  shots: number;
  counts: Record<string, number>;
  probabilities: Record<string, number>;
  num_qubits: number;
}
export const apiIBMRunCircuit = (
  ibm_token: string,
  backend_name: string,
  qasm_str: string,
  shots: number,
) => post<IBMRunCircuitResponse>("/api/ibm/run-circuit", { ibm_token, backend_name, qasm_str, shots });

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
  custom_pauli_str?: string | null;
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
  qasm_ansatz_str?: string;
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

// ─── Admin key validation (owner-only) ────────────────────────────────────────
export const apiAdminValidateKey = (service: string, api_key: string) =>
  authPost<{ service: string; valid: boolean; hint: string }>("/api/admin/validate-key", { service, api_key });

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

// ── Credit Risk Analysis ───────────────────────────────────────────────────

export interface CreditRiskObligorInput {
  name: string;
  ticker: string;
  sector: string;
  sp_rating: string;
  loan_usd: number;
  fico_score?: number;
  pd_override?: number;
  lgd_override?: number;
  rho_override?: number;
}

export interface CreditRiskObligorResult {
  name: string;
  ticker: string;
  sp_rating: string;
  sector: string;
  loan_usd: number;
  lgd_pct: number;
  lgd_usd: number;
  pd_base_pct: number;
  pd_adj_pct: number;
  rho: number;
  el_own_usd: number;
}

export interface CreditRiskRequest {
  obligors?: CreditRiskObligorInput[];
  use_presets?: boolean;
  confidence?: number;
  horizon_years?: number;
  stress_multiplier?: number;
  use_quantum?: boolean;
  n_z?: number;
  shots?: number;
}

export interface CreditRiskSource {
  title: string;
  url: string;
  description: string;
}

export interface CreditRiskResponse {
  obligors: CreditRiskObligorResult[];
  total_exposure_usd: number;
  confidence: number;
  horizon_years: number;
  stress_multiplier: number;
  mc: {
    expected_loss_usd: number;
    var_usd: number;
    cvar_usd: number;
    paths: number;
  };
  quantum: {
    used: boolean;
    el_usd: number | null;
    cvar_usd: number | null;
    circuit_info: Record<string, unknown> | null;
    error: string | null;
    available: boolean;
  };
  histogram: Array<{ loss_usd: number; probability: number; label: string }>;
  percentile_table: Array<{ label: string; loss_usd: number }>;
  multi_default_prob: number;
  sources: CreditRiskSource[];
}

export const apiCreditRiskPresets = () =>
  get<{ presets: CreditRiskObligorInput[] }>("/api/credit-risk/presets");

export const apiCreditRiskAnalyze = (req: CreditRiskRequest) =>
  post<CreditRiskResponse>("/api/credit-risk/analyze", req);

// ─── Budgeting ─────────────────────────────────────────────────────────────
// Authenticated, Supabase-persisted (one budget per user, upserted).

export interface BudgetItem {
  id: string;
  name: string;
  amount: number;
}

export interface BudgetCategory {
  id: string;
  emoji: string;
  label: string;
  items: BudgetItem[];
}

export interface SaveBudgetRequest {
  income: number;
  categories: BudgetCategory[];
}

export interface BudgetPlanResponse {
  user_id: string;
  income: number;
  categories: BudgetCategory[];
  category_totals: Record<string, number>;
  total_spend: number;
  surplus: number;
  needs_pct: number;
  wants_pct: number;
  savings_pct: number;
  targets: { needs: number; wants: number; savings: number };
  verdicts: { needs: "over" | "under" | "on_track"; wants: "over" | "under" | "on_track"; savings: "over" | "under" | "on_track" };
}

export const apiBudgetingGet = () => authGet<BudgetPlanResponse>("/api/budgeting/plan");
export const apiBudgetingSave = (req: SaveBudgetRequest) =>
  authPost<BudgetPlanResponse>("/api/budgeting/plan", req);

// ─── Retirement ──────────────────────────────────────────────────────────────
// Authenticated, Supabase-persisted (one plan per user, upserted). Phase 1
// scope: fixed-rate compounding growth. Phase 4 adds withdrawal/decumulation
// risk analysis (tax brackets, Roth/Traditional split, Monte Carlo over real
// historical S&P 500 sequences) via a separate stateless `risk-analysis` call.

export interface SaveRetirementPlanRequest {
  plan_name?: string;
  current_age: number;
  retirement_age: number;
  current_savings: number;
  monthly_contribution: number;
  expected_return_rate: number;
  inflation_rate?: number;
  retirement_goal?: string;
  roth_pct?: number;
  life_expectancy?: number;
  withdrawal_rate_pct?: number;
}

export interface RetirementSeriesPoint {
  age: number;
  balance: number;
}

export interface RetirementPlanResponse {
  user_id: string;
  plan_name: string;
  current_age: number;
  retirement_age: number;
  current_savings: number;
  monthly_contribution: number;
  expected_return_rate: number;
  inflation_rate: number;
  retirement_goal: string;
  roth_pct: number;
  life_expectancy: number;
  withdrawal_rate_pct: number;
  series: RetirementSeriesPoint[];
  late_start_series: RetirementSeriesPoint[];
  late_start_age: number;
  years: number;
  projected_balance: number;
  late_start_balance: number;
  total_contributed: number;
  total_growth: number;
  growth_multiple: number | null;
}

export const apiRetirementGet = () => authGet<RetirementPlanResponse>("/api/retirement/plan");
export const apiRetirementSave = (req: SaveRetirementPlanRequest) =>
  authPost<RetirementPlanResponse>("/api/retirement/plan", req);

// ─── Retirement Risk Analysis (Phase 4) ───────────────────────────────────────
// Stateless — a Monte Carlo block-bootstrap over real historical S&P 500 (SPY)
// annual returns, simplified 2024 federal-single-filer tax brackets on the
// Traditional portion of each year's withdrawal. Educational only.

export interface RetirementRiskAnalysisRequest {
  starting_balance: number;
  roth_pct?: number;
  withdrawal_rate_pct?: number;
  inflation_rate_pct?: number;
  horizon_years?: number;
  n_simulations?: number;
}

export interface RetirementPercentileTimelinePoint {
  year: number;
  p10: number;
  median: number;
  p90: number;
}

export interface RetirementRiskAnalysisResponse {
  success_rate_pct: number;
  median_ending_balance: number;
  worst_case_ending_balance: number;
  best_case_ending_balance: number;
  median_lifetime_taxes_paid: number;
  percentile_timeline: RetirementPercentileTimelinePoint[];
  n_simulations: number;
  data_source: string;
  initial_annual_withdrawal: number;
  initial_taxable_withdrawal: number;
  initial_tax_free_withdrawal: number;
  initial_year_tax: number;
}

export const apiRetirementRiskAnalysis = (req: RetirementRiskAnalysisRequest) =>
  authPost<RetirementRiskAnalysisResponse>("/api/retirement/risk-analysis", req);

// ─── Credit Behavior Simulator ───────────────────────────────────────────────
// Authenticated. `simulate` is stateless (no persistence); `profile` GET/POST
// saves a named scenario + its last result for next visit. Educational only —
// not a real credit-bureau scoring algorithm (see `disclaimer` in the response).

export type CreditSimPaymentBehavior = "on_time" | "minimum_only" | "missed";

export interface CreditSimMonth {
  payment_behavior: CreditSimPaymentBehavior;
  utilization_pct: number;
  new_account_opened?: boolean;
}

export interface CreditSimRequest {
  starting_fico: number;
  monthly_income: number;
  monthly_debt: number;
  months: CreditSimMonth[];
}

export interface CreditSimTrajectoryPoint {
  month: number;
  score: number;
  factor_breakdown: Record<string, number>;
}

export interface CreditSimResponse {
  trajectory: CreditSimTrajectoryPoint[];
  starting_fico: number;
  ending_fico: number;
  dti: number | null;
  disclaimer: string;
  tips: string[];
}

export const apiCreditSimSimulate = (req: CreditSimRequest) =>
  authPost<CreditSimResponse>("/api/credit-sim/simulate", req);

export interface SaveCreditSimProfileRequest {
  starting_fico: number;
  monthly_income: number;
  monthly_debt: number;
  behavior_assumptions: Record<string, unknown>;
  last_trajectory?: CreditSimResponse | null;
}

export interface CreditSimProfileResponse {
  user_id: string;
  starting_fico: number;
  monthly_income: number;
  monthly_debt: number;
  behavior_assumptions: Record<string, unknown>;
  last_trajectory: CreditSimResponse | null;
}

export const apiCreditSimGetProfile = () => authGet<CreditSimProfileResponse>("/api/credit-sim/profile");
export const apiCreditSimSaveProfile = (req: SaveCreditSimProfileRequest) =>
  authPost<CreditSimProfileResponse>("/api/credit-sim/profile", req);

// ─── Home Planning ────────────────────────────────────────────────────────────
// Authenticated. `simulate` is stateless; `plan` GET/POST saves a single
// current plan per user (option_a/option_b + shared utilities), matching the
// Budgeting/Retirement upsert-one-row pattern.

export type HomePlanningType = "home" | "apartment" | "dorm" | "mobile_home";

export interface HomeBuyInputs {
  purchase_price: number;
  down_payment_pct?: number;
  mortgage_rate_pct?: number;
  term_years?: number;
  property_tax_rate_pct?: number;
  annual_insurance?: number;
  hoa_monthly?: number;
  closing_costs_pct?: number;
}

export interface ApartmentInputs {
  monthly_rent: number;
  deposit_multiplier?: number;
  renters_insurance_monthly?: number;
}

export interface DormInputs {
  cost_per_semester: number;
  semesters_per_year?: number;
  meal_plan_included?: boolean;
}

export interface MobileHomeInputs {
  purchase_price: number;
  down_payment_pct?: number;
  loan_rate_pct?: number;
  term_years?: number;
  lot_rent_monthly?: number;
  annual_insurance?: number;
}

export interface HomePlanningOption {
  type: HomePlanningType;
  inputs: HomeBuyInputs | ApartmentInputs | DormInputs | MobileHomeInputs | Record<string, unknown>;
}

export interface HomePlanningEvaluation {
  type: HomePlanningType;
  monthly_total: number;
  upfront_cost: number;
  utilities_total: number;
  grand_total_monthly: number;
  [key: string]: unknown;
}

export interface HomePlanningTimelinePoint {
  year: number;
  cumulative_buy_spend: number;
  cumulative_rent_spend: number;
  home_value: number;
  net_buy_cost: number;
  net_rent_cost: number;
}

export interface HomePlanningComparison {
  timeline: HomePlanningTimelinePoint[];
  breakeven_year: number | null;
  upfront_comparison: { buy: number; rent: number };
  monthly_comparison: { buy: number; rent: number };
}

export interface HomePlanningSimulateRequest {
  option_a?: HomePlanningOption | null;
  option_b?: HomePlanningOption | null;
  utilities?: Record<string, number>;
  horizon_years?: number;
  appreciation_rate_pct?: number;
  selling_cost_pct?: number;
}

export interface HomePlanningSimulateResponse {
  option_a?: HomePlanningEvaluation;
  option_b?: HomePlanningEvaluation;
  comparison?: HomePlanningComparison;
  utilities_used: Record<string, number>;
}

export interface HomePlanningUtilityDefault {
  label: string;
  default_monthly: number;
}

export const apiHomePlanningDefaults = () =>
  get<{ utilities: Record<string, HomePlanningUtilityDefault> }>("/api/home-planning/defaults");

export const apiHomePlanningSimulate = (req: HomePlanningSimulateRequest) =>
  authPost<HomePlanningSimulateResponse>("/api/home-planning/simulate", req);

export interface SaveHomePlanRequest {
  option_a?: HomePlanningOption | null;
  option_b?: HomePlanningOption | null;
  utilities?: Record<string, number>;
  comparison_settings?: Record<string, unknown>;
}

export interface HomePlanResponse {
  user_id: string;
  option_a: HomePlanningOption | null;
  option_b: HomePlanningOption | null;
  utilities: Record<string, number>;
  comparison_settings: Record<string, unknown>;
}

export const apiHomePlanningGet = () => authGet<HomePlanResponse>("/api/home-planning/plan");
export const apiHomePlanningSave = (req: SaveHomePlanRequest) =>
  authPost<HomePlanResponse>("/api/home-planning/plan", req);

// ─── Debt Management ──────────────────────────────────────────────────────────
// Authenticated (except the minimum-payment hint, a stateless calc helper).
// `simulate` is stateless; `plan` GET/POST saves a single current plan per
// user (the full debt list + extra monthly payment budget).

export type DebtType = "student_loan" | "credit_card" | "personal_loan" | "business_loan";
export type DebtStrategy = "minimum_only" | "snowball" | "avalanche";

export interface DebtEntry {
  id: string;
  name: string;
  type: DebtType;
  balance: number;
  apr_pct: number;
  minimum_payment: number;
}

export interface DebtPlanSimulateRequest {
  debts: DebtEntry[];
  extra_monthly_payment?: number;
}

export interface DebtTimelinePoint {
  month: number;
  total_remaining_balance: number;
}

export interface DebtPayoffOrderEntry {
  id: string;
  payoff_month: number | null;
}

export interface DebtStrategyResult {
  strategy: DebtStrategy;
  months_to_debt_free: number | null;
  total_interest_paid: number;
  payoff_order: DebtPayoffOrderEntry[];
  timeline: DebtTimelinePoint[];
  hit_cap: boolean;
}

export interface DebtStrategySavings {
  interest_saved: number;
  months_saved: number | null;
}

export interface DebtPlanSimulateResponse {
  strategies: Record<DebtStrategy, DebtStrategyResult>;
  summary: {
    snowball_vs_minimum: DebtStrategySavings;
    avalanche_vs_minimum: DebtStrategySavings;
    avalanche_vs_snowball: DebtStrategySavings;
  };
}

export const apiDebtManagementSimulate = (req: DebtPlanSimulateRequest) =>
  authPost<DebtPlanSimulateResponse>("/api/debt-management/simulate", req);

export const apiDebtManagementMinimumPaymentHint = (type: DebtType, balance: number, apr_pct: number) =>
  post<{ minimum_payment_hint: number }>("/api/debt-management/minimum-payment-hint", { type, balance, apr_pct });

export interface SaveDebtPlanRequest {
  debts: DebtEntry[];
  extra_monthly_payment?: number;
}

export interface DebtPlanResponse {
  user_id: string;
  debts: DebtEntry[];
  extra_monthly_payment: number;
}

export const apiDebtManagementGet = () => authGet<DebtPlanResponse>("/api/debt-management/plan");
export const apiDebtManagementSave = (req: SaveDebtPlanRequest) =>
  authPost<DebtPlanResponse>("/api/debt-management/plan", req);
