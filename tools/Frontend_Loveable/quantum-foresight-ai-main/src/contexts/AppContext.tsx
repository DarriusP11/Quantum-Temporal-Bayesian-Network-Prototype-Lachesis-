/**
 * AppContext.tsx — Global state shared between the sidebar and all tabs.
 * Mirrors the Streamlit sidebar session-state keys.
 */
import { createContext, useContext, useState, useCallback, ReactNode } from "react";

// ── Language support ────────────────────────────────────────────────────────
export interface LanguageOption { code: string; label: string; native: string; }

export const SUPPORTED_LANGUAGES: LanguageOption[] = [
  { code: "English",    label: "English",    native: "English" },
  { code: "Spanish",    label: "Spanish",    native: "Español" },
  { code: "French",     label: "French",     native: "Français" },
  { code: "German",     label: "German",     native: "Deutsch" },
  { code: "Italian",    label: "Italian",    native: "Italiano" },
  { code: "Portuguese", label: "Portuguese", native: "Português" },
  { code: "Dutch",      label: "Dutch",      native: "Nederlands" },
  { code: "Russian",    label: "Russian",    native: "Русский" },
  { code: "Chinese (Simplified)",  label: "Chinese (Simplified)",  native: "中文（简体）" },
  { code: "Chinese (Traditional)", label: "Chinese (Traditional)", native: "中文（繁體）" },
  { code: "Japanese",   label: "Japanese",   native: "日本語" },
  { code: "Korean",     label: "Korean",     native: "한국어" },
  { code: "Arabic",     label: "Arabic",     native: "العربية" },
  { code: "Hindi",      label: "Hindi",      native: "हिन्दी" },
  { code: "Bengali",    label: "Bengali",    native: "বাংলা" },
  { code: "Turkish",    label: "Turkish",    native: "Türkçe" },
  { code: "Vietnamese", label: "Vietnamese", native: "Tiếng Việt" },
  { code: "Thai",       label: "Thai",       native: "ไทย" },
  { code: "Indonesian", label: "Indonesian", native: "Bahasa Indonesia" },
  { code: "Malay",      label: "Malay",      native: "Bahasa Melayu" },
  { code: "Polish",     label: "Polish",     native: "Polski" },
  { code: "Swedish",    label: "Swedish",    native: "Svenska" },
  { code: "Norwegian",  label: "Norwegian",  native: "Norsk" },
  { code: "Danish",     label: "Danish",     native: "Dansk" },
  { code: "Finnish",    label: "Finnish",    native: "Suomi" },
  { code: "Romanian",   label: "Romanian",   native: "Română" },
  { code: "Greek",      label: "Greek",      native: "Ελληνικά" },
  { code: "Hungarian",  label: "Hungarian",  native: "Magyar" },
  { code: "Czech",      label: "Czech",      native: "Čeština" },
  { code: "Slovak",     label: "Slovak",     native: "Slovenčina" },
  { code: "Ukrainian",  label: "Ukrainian",  native: "Українська" },
  { code: "Bulgarian",  label: "Bulgarian",  native: "Български" },
  { code: "Croatian",   label: "Croatian",   native: "Hrvatski" },
  { code: "Serbian",    label: "Serbian",    native: "Српски" },
  { code: "Lithuanian", label: "Lithuanian", native: "Lietuvių" },
  { code: "Latvian",    label: "Latvian",    native: "Latviešu" },
  { code: "Estonian",   label: "Estonian",   native: "Eesti" },
  { code: "Hebrew",     label: "Hebrew",     native: "עברית" },
  { code: "Persian",    label: "Persian",    native: "فارسی" },
  { code: "Urdu",       label: "Urdu",       native: "اردو" },
  { code: "Swahili",    label: "Swahili",    native: "Kiswahili" },
  { code: "Tamil",      label: "Tamil",      native: "தமிழ்" },
  { code: "Telugu",     label: "Telugu",     native: "తెలుగు" },
  { code: "Kannada",    label: "Kannada",    native: "ಕನ್ನಡ" },
  { code: "Marathi",    label: "Marathi",    native: "मराठी" },
  { code: "Gujarati",   label: "Gujarati",   native: "ગુજરાતી" },
  { code: "Punjabi",    label: "Punjabi",    native: "ਪੰਜਾਬੀ" },
  { code: "Afrikaans",  label: "Afrikaans",  native: "Afrikaans" },
  { code: "Catalan",    label: "Catalan",    native: "Català" },
  { code: "Welsh",      label: "Welsh",      native: "Cymraeg" },
];

// ── Gate step type ─────────────────────────────────────────────────────────────
export type GateChoice = "None"|"H"|"X"|"Y"|"Z"|"RX"|"RY"|"RZ"|"S"|"T";

export interface GateStepConfig {
  q0: GateChoice; q0_angle: number;
  q1: GateChoice; q1_angle: number;
  q2: GateChoice; q2_angle: number;
  q3: GateChoice; q3_angle: number;
  cnot_01: boolean; cnot_12: boolean; cnot_23: boolean;
}

const defaultStep = (): GateStepConfig => ({
  q0: "None", q0_angle: 0.0,
  q1: "None", q1_angle: 0.0,
  q2: "None", q2_angle: 0.0,
  q3: "None", q3_angle: 0.0,
  cnot_01: false, cnot_12: false, cnot_23: false,
});

// ── Noise config ───────────────────────────────────────────────────────────────
export interface NoiseConfig {
  enable_depolarizing: boolean;
  pdep0: number; pdep1: number; pdep2: number;
  enable_amplitude_damping: boolean;
  pamp0: number; pamp1: number; pamp2: number;
  enable_phase_damping: boolean;
  pph0: number; pph1: number; pph2: number;
  enable_cnot_noise: boolean;
  pcnot0: number; pcnot1: number; pcnot2: number;
}

// ── Finance config ─────────────────────────────────────────────────────────────
export interface FinanceConfig {
  tickers: string;
  lookback_days: number;
  portfolio_value: number;
  confidence_level: number;
  var_horizon: number;
  mc_sims: number;
  volatility_threshold: number;
  apply_macro_stress: boolean;
  demo_mode: boolean;
  per_share: boolean;
  show_position: boolean;
}

// ── Full app state ─────────────────────────────────────────────────────────────
export interface AppState {
  // Quantum
  num_qubits: number;
  shots: number;
  use_seed: boolean;
  seed_val: number;
  step0: GateStepConfig;
  step1: GateStepConfig;
  step2: GateStepConfig;
  noise: NoiseConfig;
  // Finance
  finance: FinanceConfig;
  // Global LLM language
  language: string;
}

const DEFAULT_STATE: AppState = {
  language: "English",
  num_qubits: 1,
  shots: 2048,
  use_seed: true,
  seed_val: 17,
  step0: { ...defaultStep(), q0: "H", q0_angle: 0.5 },
  step1: defaultStep(),
  step2: defaultStep(),
  noise: {
    enable_depolarizing: false,  pdep0: 0.01,  pdep1: 0.02,  pdep2: 0.02,
    enable_amplitude_damping: false, pamp0: 0.0,   pamp1: 0.02,  pamp2: 0.02,
    enable_phase_damping: false, pph0: 0.04,   pph1: 0.05,   pph2: 0.05,
    enable_cnot_noise: false,    pcnot0: 0.02, pcnot1: 0.02, pcnot2: 0.02,
  },
  finance: {
    tickers: "SPY,QQQ,AAPL",
    lookback_days: 365,
    portfolio_value: 100000,
    confidence_level: 0.95,
    var_horizon: 10,
    mc_sims: 50000,
    volatility_threshold: 0.30,
    apply_macro_stress: false,
    demo_mode: true,
    per_share: false,
    show_position: false,
  },
};

// ── Context type ───────────────────────────────────────────────────────────────
interface AppContextValue {
  state: AppState;
  setNumQubits: (n: number) => void;
  setShots: (n: number) => void;
  setUseSeed: (b: boolean) => void;
  setSeedVal: (n: number) => void;
  setStep: (idx: 0|1|2, s: GateStepConfig) => void;
  setNoise: (n: NoiseConfig) => void;
  setFinance: (f: FinanceConfig) => void;
  setLanguage: (lang: string) => void;
  resetToDefaults: () => void;
  /** Build the API request body from current state (for /api/quantum/simulate) */
  buildQuantumRequest: () => object;
}

const AppContext = createContext<AppContextValue | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AppState>(DEFAULT_STATE);

  const setNumQubits = useCallback((n: number) =>
    setState(s => ({ ...s, num_qubits: n })), []);
  const setShots = useCallback((n: number) =>
    setState(s => ({ ...s, shots: n })), []);
  const setUseSeed = useCallback((b: boolean) =>
    setState(s => ({ ...s, use_seed: b })), []);
  const setSeedVal = useCallback((n: number) =>
    setState(s => ({ ...s, seed_val: n })), []);
  const setStep = useCallback((idx: 0|1|2, step: GateStepConfig) =>
    setState(s => {
      const steps: [GateStepConfig, GateStepConfig, GateStepConfig] = [s.step0, s.step1, s.step2];
      steps[idx] = step;
      return { ...s, step0: steps[0], step1: steps[1], step2: steps[2] };
    }), []);
  const setNoise = useCallback((n: NoiseConfig) =>
    setState(s => ({ ...s, noise: n })), []);
  const setFinance = useCallback((f: FinanceConfig) =>
    setState(s => ({ ...s, finance: f })), []);
  const setLanguage = useCallback((lang: string) =>
    setState(s => ({ ...s, language: lang })), []);
  const resetToDefaults = useCallback(() => setState(DEFAULT_STATE), []);

  const buildQuantumRequest = useCallback(() => {
    const { num_qubits, shots, use_seed, seed_val, step0, step1, step2, noise } = state;
    return {
      num_qubits,
      shots,
      seed: use_seed ? seed_val : undefined,
      step0, step1, step2,
      noise: {
        enable_depolarizing: noise.enable_depolarizing,
        depolarizing_prob: noise.pdep0,
        enable_amplitude_damping: noise.enable_amplitude_damping,
        amplitude_damping_prob: noise.pamp0,
        enable_phase_damping: noise.enable_phase_damping,
        phase_damping_prob: noise.pph0,
        enable_cnot_noise: noise.enable_cnot_noise,
        cnot_noise_prob: noise.pcnot0,
      },
    };
  }, [state]);

  return (
    <AppContext.Provider value={{
      state, setNumQubits, setShots, setUseSeed, setSeedVal,
      setStep, setNoise, setFinance, setLanguage, resetToDefaults, buildQuantumRequest,
    }}>
      {children}
    </AppContext.Provider>
  );
}

export function useAppContext() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useAppContext must be used inside AppProvider");
  return ctx;
}
