/**
 * features.ts — reversible feature flags.
 *
 * Quantum tabs/components/routes are kept fully intact in the codebase;
 * this flag only controls whether the "Quantum" section is reachable in
 * the UI. Set VITE_ENABLE_QUANTUM=false in .env to hide it.
 */
export const FEATURES = {
  quantum: import.meta.env.VITE_ENABLE_QUANTUM !== "false",
};
