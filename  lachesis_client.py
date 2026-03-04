# lachesis_client.py
import os, time, random

class LachesisClient:
    def __init__(self, endpoint=None, api_key=None, timeout=10):
        self.endpoint = endpoint or os.getenv("LACHESIS_ENDPOINT")
        self.api_key  = api_key  or os.getenv("LACHESIS_API_KEY")
        self.timeout  = timeout

    @property
    def is_configured(self):
        return bool(self.endpoint and self.api_key)

    # ---- Mocked calls (swap impl when backend is ready) ----
    def health(self):
        if not self.is_configured:
            return {"status":"mock","detail":"No endpoint/key; using mock."}
        # TODO: real ping
        return {"status":"ok","detail":"Connected (stub)"}

    def get_scenarios(self):
        # TODO: real API fetch
        return [
            {"id":"bell_prep","label":"Bell prep","desc":"Create entanglement then sample under noise."},
            {"id":"dephasing_stress","label":"Dephasing stress","desc":"Phase damping ramp on superposition."},
            {"id":"amp_relax","label":"Amplitude relaxation","desc":"T1‑like decay from |1> to |0>."},
        ]

    def forecast(self, context):
        """Return mock foresight: predicted P(bitstrings) & confidence."""
        time.sleep(0.2)
        keys = context.get("keys", ["0","1"])
        probs = [random.random() for _ in keys]
        s = sum(probs); probs = [p/s for p in probs]
        conf = 0.6 + 0.4*random.random()
        return {"pred": dict(zip(keys, probs)), "confidence": conf}
