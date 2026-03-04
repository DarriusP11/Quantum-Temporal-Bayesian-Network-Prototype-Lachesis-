from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import streamlit as st

OWNER_EMAIL = "darriusperson@gmail.com"
API_KEY_SPECS = {
    "openai": {
        "session_key": "openai_api_key",
        "secret_key": "openai_api_key",
        "env_key": "QTBN_OPENAI_API_KEY",
    },
    "fred": {
        "session_key": "fred_api_key",
        "secret_key": "fred_api_key",
        "env_key": "QTBN_FRED_API_KEY",
    },
    "perplexity": {
        "session_key": "perplexity_api_key",
        "secret_key": "perplexity_api_key",
        "env_key": "QTBN_PERPLEXITY_API_KEY",
    },
    "voice_openai": {
        "session_key": "voice_openai_api_key",
        "secret_key": "lachesis_voice_api_key",
        "env_key": "QTBN_LACHESIS_VOICE_API_KEY",
    },
    "voice_elevenlabs": {
        "session_key": "voice_elevenlabs_api_key",
        "secret_key": "elevenlabs_api_key",
        "env_key": "QTBN_ELEVENLABS_API_KEY",
    },
}
SESSION_API_KEY_FIELDS = [
    "openai_api_key",
    "fred_api_key",
    "perplexity_api_key",
    "voice_openai_api_key",
    "voice_elevenlabs_api_key",
    "portfolio_ss_api_key",
    "lachesis_voice_api_key",
    "financial_lachesis_api_key",
    "insider_lachesis_api_key",
    "sentiment_lacheiss_api_key",
    "sentiment_perplexity_api_key",
]


def normalize_email(email_text: str) -> str:
    return (email_text or "").strip().lower()


def is_owner_user() -> bool:
    if not bool(st.session_state.get("auth_ok")):
        return False

    # Trust explicit owner flag when already established for this auth session.
    if bool(st.session_state.get("auth_is_owner", False)):
        return True

    email = normalize_email(
        st.session_state.get("auth_email_normalized", "")
        or st.session_state.get("auth_email", "")
    )
    is_owner = email == OWNER_EMAIL
    if email:
        st.session_state["auth_email_normalized"] = email
    st.session_state["auth_is_owner"] = is_owner
    return is_owner


def resolve_api_key(service: str) -> str:
    spec = API_KEY_SPECS.get((service or "").strip())
    if not spec:
        return ""

    if is_owner_user():
        admin_input_key = f"admin_{spec['session_key']}_input"
        admin_val = st.session_state.get(admin_input_key, "")
        if isinstance(admin_val, str) and admin_val.strip():
            # Keep canonical override in sync with Admin input value.
            st.session_state[spec["session_key"]] = admin_val.strip()
            return admin_val.strip()

        session_val = st.session_state.get(spec["session_key"], "")
        if isinstance(session_val, str) and session_val.strip():
            return session_val.strip()

    try:
        secret_val = st.secrets.get(spec["secret_key"], "")
        if isinstance(secret_val, str) and secret_val.strip():
            return secret_val.strip()
    except Exception:
        pass

    # Fallback to project-local secrets.toml when Streamlit secrets are unavailable.
    try:
        local_secrets = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
        if local_secrets.exists():
            for raw_line in local_secrets.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() != spec["secret_key"]:
                    continue
                candidate = value.strip()
                if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in {"'", '"'}:
                    candidate = candidate[1:-1]
                if candidate.strip():
                    return candidate.strip()
    except Exception:
        pass

    env_val = os.getenv(spec["env_key"], "")
    if isinstance(env_val, str) and env_val.strip():
        return env_val.strip()
    return ""


def clear_session_api_key_overrides() -> None:
    for key in SESSION_API_KEY_FIELDS:
        if key in st.session_state:
            st.session_state[key] = ""
    for key in list(st.session_state.keys()):
        if str(key).endswith("_api_key"):
            st.session_state[key] = ""


def clear_auth_session() -> None:
    clear_session_api_key_overrides()
    for key in (
        "auth_ok",
        "auth_source",
        "auth_email_normalized",
        "auth_is_owner",
        "auth_email",
        "auth_password",
        "auth_password_confirm",
        "auth_username",
    ):
        if key in st.session_state:
            st.session_state.pop(key, None)
    for key in list(st.session_state.keys()):
        if str(key).startswith("admin_") and str(key).endswith("_input"):
            st.session_state.pop(key, None)


def apply_qtbn_purple_theme() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Fraunces:wght@600;700&display=swap');

:root {
  --qtbn-bg-0: #120a1f;
  --qtbn-bg-1: #1b0f2d;
  --qtbn-bg-2: #2b1446;
  --qtbn-accent: #a855f7;
  --qtbn-accent-2: #f472b6;
  --qtbn-text: #f8d76b;
  --qtbn-muted: #e8cf8a;
  --qtbn-card: rgba(36, 22, 62, 0.82);
  --qtbn-border: rgba(168, 85, 247, 0.22);
}

html, body, [class*="css"]  {
  font-family: "Space Grotesk", system-ui, sans-serif;
  color: var(--qtbn-text);
  text-rendering: geometricPrecision;
}

.stApp {
  background: radial-gradient(1200px 1200px at 10% 5%, #2b1550 0%, #1a0f2c 45%, #0d0717 100%);
}

h1, h2, h3, h4, h5, h6 {
  font-family: "Fraunces", "Space Grotesk", serif;
  color: var(--qtbn-text);
  letter-spacing: 0.2px;
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #1a0f2c 0%, #120a1f 100%);
  border-right: 1px solid var(--qtbn-border);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5,
section[data-testid="stSidebar"] h6 {
  color: #b8d6ff;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] div {
  color: #b8d6ff;
}

.stButton > button {
  background: linear-gradient(135deg, var(--qtbn-accent), var(--qtbn-accent-2));
  color: white;
  border: none;
  border-radius: 999px;
  padding: 0.5rem 1.2rem;
  font-weight: 600;
  box-shadow: 0 10px 20px rgba(168, 85, 247, 0.35);
}

.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 12px 24px rgba(244, 114, 182, 0.4);
}

.stTabs [data-baseweb="tab-list"] {
  gap: 0;
  background: rgba(18, 10, 31, 0.5);
  padding: 0.2rem;
  border-radius: 999px;
  border: 1px solid var(--qtbn-border);
}

.stTabs [data-baseweb="tab"] {
  background: transparent;
  border-radius: 999px;
  color: var(--qtbn-muted);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  padding: 0.35rem 0.8rem;
  margin: 0;
  border-right: 1px solid rgba(232, 207, 138, 0.55);
}

.stTabs [aria-selected="true"] {
  background: rgba(168, 85, 247, 0.25);
  color: var(--qtbn-text);
}

.stTabs [data-baseweb="tab"]:last-of-type {
  border-right: none;
}

.stMetric {
  background: var(--qtbn-card);
  border: 1px solid var(--qtbn-border);
  border-radius: 16px;
  padding: 0.75rem;
  box-shadow: 0 10px 30px rgba(12, 6, 24, 0.45);
}

.stExpander {
  border: 1px solid var(--qtbn-border);
  border-radius: 16px;
  background: var(--qtbn-card);
}

.stTextInput input, .stTextArea textarea, .stNumberInput input, .stSelectbox div, .stMultiSelect div {
  border-radius: 12px;
  border: 1px solid rgba(168, 85, 247, 0.35);
}

[data-testid="stToggle"], [data-testid="stToggle"] * {
  color: #111111 !important;
}

[data-testid="stToggle"] {
  background: #ffffff !important;
  border-radius: 999px;
  padding: 0.2rem 0.4rem;
}

.qtbn-lachesis-card {
  display: flex;
  gap: 1rem;
  align-items: center;
  background: var(--qtbn-card);
  border: 1px solid var(--qtbn-border);
  border-radius: 18px;
  padding: 0.9rem 1.1rem;
  margin: 0.5rem 0 0.8rem;
  box-shadow: 0 12px 28px rgba(12, 6, 24, 0.45);
}

.qtbn-avatar {
  width: 52px;
  height: 52px;
  border-radius: 50%;
  background: radial-gradient(circle at 30% 30%, #fcd34d, #a855f7 60%, #6b21a8 100%);
  box-shadow: 0 10px 20px rgba(168, 85, 247, 0.35);
  flex: 0 0 auto;
}

.qtbn-lachesis-title {
  font-weight: 700;
  font-size: 1.05rem;
}

.qtbn-lachesis-subtitle {
  color: var(--qtbn-muted);
  font-size: 0.9rem;
}

.qtbn-auth-wrap {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  padding-top: 0;
}

.qtbn-auth-card {
  width: min(420px, 92vw);
  height: auto;
  background: linear-gradient(160deg, rgba(88, 28, 135, 0.9), rgba(43, 20, 70, 0.95));
  border: 1px solid rgba(168, 85, 247, 0.4);
  border-radius: 28px;
  padding: 18px 24px 20px;
  box-shadow: 0 20px 40px rgba(12, 6, 24, 0.55);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  gap: 0.45rem;
  margin-top: 0;
}

.qtbn-auth-title {
  font-family: "Fraunces", "Space Grotesk", serif;
  color: #f8d76b;
  font-size: 2.2rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-top: 0.15rem;
  margin-bottom: 0.2rem;
  width: 210px;
  text-align: center;
}

.qtbn-auth-logo-box {
  width: 170px;
  height: 170px;
  border-radius: 26px;
  background: transparent;
  border: none;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0;
  overflow: hidden;
}

.qtbn-auth-logo-box img {
  display: block;
  margin: 0 auto;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_lachesis_voice_panel(section_key: str) -> None:
    st.markdown(
        """
<div class="qtbn-lachesis-card">
  <div class="qtbn-avatar" aria-hidden="true"></div>
  <div>
    <div class="qtbn-lachesis-title">Lachesis Companion</div>
    <div class="qtbn-lachesis-subtitle">Voice-ready narration panel</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    provider_key = f"lachesis_voice_provider_{section_key}"
    enable_key = f"lachesis_voice_enabled_{section_key}"
    voice_name_key = f"lachesis_voice_name_{section_key}"
    model_key = f"lachesis_voice_model_{section_key}"
    el_voice_id_key = f"lachesis_eleven_voice_id_{section_key}"
    el_model_key = f"lachesis_eleven_model_{section_key}"
    if "lachesis_voice_provider" not in st.session_state:
        st.session_state["lachesis_voice_provider"] = ""
    if "lachesis_voice_enabled" not in st.session_state:
        st.session_state["lachesis_voice_enabled"] = False
    if "lachesis_voice_name" not in st.session_state:
        st.session_state["lachesis_voice_name"] = "alloy"
    if "lachesis_voice_model" not in st.session_state:
        st.session_state["lachesis_voice_model"] = "gpt-4o-mini-tts"
    if "lachesis_eleven_voice_id" not in st.session_state:
        st.session_state["lachesis_eleven_voice_id"] = "EXAVITQu4vr4xnSDxMaL"
    if "lachesis_eleven_model" not in st.session_state:
        st.session_state["lachesis_eleven_model"] = "eleven_multilingual_v2"

    provider = st.selectbox(
        "Voice provider",
        ["(none)", "OpenAI", "ElevenLabs", "Azure", "Other"],
        index=0,
        key=provider_key,
    )
    enabled = st.checkbox(
        "Enable voice playback",
        value=st.session_state.get("lachesis_voice_enabled", False),
        key=enable_key,
    )
    voice_name = st.selectbox(
        "Voice",
        ["alloy", "nova", "shimmer", "echo", "fable", "onyx"],
        index=0,
        key=voice_name_key,
    )
    model = st.text_input(
        "Voice model",
        value=st.session_state.get("lachesis_voice_model", "gpt-4o-mini-tts"),
        key=model_key,
        help="OpenAI TTS model name.",
    )
    if provider == "ElevenLabs":
        st.text_input(
            "ElevenLabs Voice ID",
            value=st.session_state.get("lachesis_eleven_voice_id", "EXAVITQu4vr4xnSDxMaL"),
            key=el_voice_id_key,
            help="Find this in your ElevenLabs dashboard (voice library).",
        )
        st.text_input(
            "ElevenLabs Model ID",
            value=st.session_state.get("lachesis_eleven_model", "eleven_multilingual_v2"),
            key=el_model_key,
            help="Common: eleven_multilingual_v2, eleven_turbo_v2.",
        )
    st.session_state["lachesis_voice_provider"] = "" if provider == "(none)" else provider
    st.session_state["lachesis_voice_enabled"] = bool(enabled)
    st.session_state["lachesis_voice_name"] = voice_name
    st.session_state["lachesis_voice_model"] = model
    st.session_state["lachesis_eleven_voice_id"] = st.session_state.get(el_voice_id_key, st.session_state["lachesis_eleven_voice_id"])
    st.session_state["lachesis_eleven_model"] = st.session_state.get(el_model_key, st.session_state["lachesis_eleven_model"])

    voice_service = ""
    if provider == "OpenAI":
        voice_service = "voice_openai"
    elif provider == "ElevenLabs":
        voice_service = "voice_elevenlabs"

    if voice_service and not resolve_api_key(voice_service):
        st.caption("Voice provider selected but API key is not configured.")
    if provider == "(none)":
        st.caption("Select a provider to enable audio playback.")


def synthesize_lachesis_audio(text: str) -> tuple[bytes | None, str | None]:
    provider = st.session_state.get("lachesis_voice_provider", "")
    enabled = bool(st.session_state.get("lachesis_voice_enabled", False))
    if not enabled or not provider or not text:
        return None, None

    try:
        import requests
    except Exception as e:
        return None, f"requests not available: {e}"

    if provider == "OpenAI":
        api_key = resolve_api_key("voice_openai")
        if not api_key:
            return None, "Missing voice API key."
        model = st.session_state.get("lachesis_voice_model", "gpt-4o-mini-tts")
        voice = st.session_state.get("lachesis_voice_name", "alloy")
        payload = {
            "model": model,
            "voice": voice,
            "input": text,
            "format": "mp3",
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(
                "https://api.openai.com/v1/audio/speech",
                headers=headers,
                json=payload,
                timeout=60,
            )
            if resp.status_code >= 300:
                return None, f"Voice API error {resp.status_code}: {resp.text[:200]}"
            return resp.content, None
        except Exception as e:
            return None, str(e)

    if provider == "ElevenLabs":
        api_key = resolve_api_key("voice_elevenlabs")
        if not api_key:
            return None, "Missing voice API key."
        voice_id = st.session_state.get("lachesis_eleven_voice_id", "EXAVITQu4vr4xnSDxMaL")
        model_id = st.session_state.get("lachesis_eleven_model", "eleven_multilingual_v2")
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.7},
        }
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        try:
            resp = requests.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers=headers,
                json=payload,
                timeout=60,
            )
            if resp.status_code >= 300:
                return None, f"ElevenLabs error {resp.status_code}: {resp.text[:200]}"
            return resp.content, None
        except Exception as e:
            return None, str(e)

    return None, "Voice provider not implemented yet."


def render_llm_disclaimer() -> None:
    st.caption(
        "LLM note: A large language model (LLM) is an AI system that generates text from patterns in data. "
        "It can hallucinate, which means it may produce confident-sounding but incorrect or fabricated "
        "information. Always verify important details."
    )


def render_auth_gate(logo_path: str | None = None) -> bool:
    if st.session_state.get("auth_ok"):
        email_norm = normalize_email(st.session_state.get("auth_email_normalized", ""))
        if not email_norm:
            email_norm = normalize_email(st.session_state.get("auth_email", ""))
            if email_norm:
                st.session_state["auth_email_normalized"] = email_norm
        st.session_state["auth_is_owner"] = is_owner_user()
        return True

    # Center the unauthenticated screen so logo + auth controls appear in the middle.
    st.markdown(
        """
<style>
[data-testid="stAppViewContainer"] .main .block-container {
  max-width: 520px !important;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding-top: 0.75rem !important;
  padding-bottom: 0.75rem !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )

    def _safe_secret(key: str) -> str:
        try:
            value = st.secrets.get(key, "")
            if isinstance(value, str) and value.strip():
                return value.strip()
        except Exception:
            pass
        # Fallback: read the project-local Streamlit secrets file directly.
        # This keeps auth working even when the app is launched from a different CWD.
        try:
            local_secrets = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
            if local_secrets.exists():
                for raw_line in local_secrets.read_text(encoding="utf-8").splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    if k.strip() != key:
                        continue
                    value = v.strip()
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                        value = value[1:-1]
                    return value.strip()
        except Exception:
            pass
        return ""

    supabase_url = (
        os.getenv("QTBN_SUPABASE_URL", "").strip()
        or _safe_secret("supabase_url")
    )
    supabase_key = (
        os.getenv("QTBN_SUPABASE_ANON_KEY", "").strip()
        or _safe_secret("supabase_anon_key")
    )
    users_path = Path(__file__).resolve().parent / "auth_users.json"

    def _load_users() -> dict:
        try:
            if users_path.exists():
                data = json.loads(users_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {}

    def _save_users(users: dict) -> bool:
        try:
            users_path.write_text(json.dumps(users, indent=2), encoding="utf-8")
            return True
        except Exception:
            return False

    def _hash_password(password_text: str, salt: str) -> str:
        return hashlib.sha256((salt + password_text).encode("utf-8")).hexdigest()

    def _verify_password(password_text: str, salt: str, digest: str) -> bool:
        return _hash_password(password_text, salt) == digest

    def _local_sign_in(email_text: str, password_text: str) -> bool:
        users = _load_users()
        record = users.get(email_text.strip().lower())
        if not isinstance(record, dict):
            return False
        salt = str(record.get("salt", ""))
        digest = str(record.get("hash", ""))
        if not salt or not digest:
            return False
        return _verify_password(password_text, salt, digest)

    def _local_sign_up(email_text: str, password_text: str, username_text: str) -> tuple[bool, str]:
        users = _load_users()
        key = email_text.strip().lower()
        if key in users:
            return False, "Account already exists for this email in local failover store."
        salt = hashlib.sha256(os.urandom(16)).hexdigest()[:16]
        users[key] = {
            "salt": salt,
            "hash": _hash_password(password_text, salt),
            "username": username_text.strip(),
        }
        if not _save_users(users):
            return False, "Failed to save account to local failover store."
        return True, "Account created in local failover store."

    def _get_supabase_client():
        if not supabase_url or not supabase_key:
            return None
        try:
            from supabase import create_client  # type: ignore
            return create_client(supabase_url, supabase_key)
        except Exception:
            return None

    st.markdown('<div class="qtbn-auth-wrap"><div class="qtbn-auth-card">', unsafe_allow_html=True)
    st.markdown('<div class="qtbn-auth-logo-box">', unsafe_allow_html=True)
    if logo_path:
        logo_file = Path(logo_path)
        if logo_file.exists():
            try:
                st.image(str(logo_file), width=170)
            except Exception:
                pass
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="qtbn-auth-title">Lachesis</div>', unsafe_allow_html=True)

    supabase_client = _get_supabase_client()
    if supabase_client is None:
        st.warning("Supabase unavailable. Using local failover auth store (`auth_users.json`).")
    else:
        st.caption("Auth provider: Supabase (with local failover enabled)")

    mode = st.radio("Mode", ["Sign in", "Sign up"], horizontal=True, key="auth_mode")
    email = st.text_input("Email", key="auth_email")
    password = st.text_input("Password", type="password", key="auth_password")
    username = None
    password2 = None
    if mode == "Sign up":
        username = st.text_input("Username", key="auth_username")
        password2 = st.text_input("Confirm password", type="password", key="auth_password_confirm")

    if mode == "Sign in":
        if st.button("Sign in"):
            if not email or not password:
                st.error("Enter both email and password.")
                st.markdown("</div></div>", unsafe_allow_html=True)
                return False

            auth_ok = False
            supabase_error = None
            if supabase_client is not None:
                try:
                    supabase_client.auth.sign_in_with_password(
                        {"email": email.strip(), "password": password}
                    )
                    auth_ok = True
                    st.session_state["auth_source"] = "supabase"
                except Exception as e:
                    supabase_error = str(e)

            if not auth_ok and _local_sign_in(email, password):
                auth_ok = True
                st.session_state["auth_source"] = "local_failover"
                if supabase_error:
                    st.warning("Supabase sign-in failed. Signed in from local failover store.")
                elif supabase_client is None:
                    st.info("Signed in from local failover store.")

            if auth_ok:
                email_norm = normalize_email(email)
                st.session_state["auth_ok"] = True
                st.session_state["auth_email_normalized"] = email_norm
                st.session_state["auth_is_owner"] = (email_norm == OWNER_EMAIL)
                st.success("Authenticated.")
                st.rerun()
            elif supabase_error:
                st.error(f"Sign-in failed in Supabase and local failover: {supabase_error}")
            else:
                st.error("Invalid email or password.")
    else:
        if st.button("Create account"):
            if not email or "@" not in email:
                st.error("Enter a valid email address.")
            elif not username or len(username.strip()) < 3:
                st.error("Username must be at least 3 characters.")
            elif not password or len(password) < 6:
                st.error("Password must be at least 6 characters.")
            elif password2 is None or password != password2:
                st.error("Passwords do not match.")
            else:
                supabase_error = None
                supabase_created = False
                if supabase_client is not None:
                    try:
                        supabase_client.auth.sign_up(
                            {
                                "email": email.strip(),
                                "password": password,
                                "options": {
                                    "data": {
                                        "username": username.strip(),
                                    }
                                },
                            }
                        )
                        supabase_created = True
                    except Exception as e:
                        supabase_error = str(e)

                if supabase_created:
                    st.success("Account created. Check your email to confirm, then sign in.")
                    st.markdown("</div></div>", unsafe_allow_html=True)
                    return False

                local_created, local_msg = _local_sign_up(email, password, username)
                if local_created:
                    if supabase_error:
                        st.warning(f"Supabase sign-up failed. {local_msg}")
                    elif supabase_client is None:
                        st.info(local_msg)
                    else:
                        st.warning(local_msg)
                    st.markdown("</div></div>", unsafe_allow_html=True)
                    return False

                if supabase_error:
                    st.error(f"Supabase sign-up failed and local fallback failed: {supabase_error} | {local_msg}")
                else:
                    st.error(local_msg)
                st.markdown("</div></div>", unsafe_allow_html=True)
                return False

    st.markdown("</div></div>", unsafe_allow_html=True)
    return False
