# app.py — SEC Filings Chatbot (Streamlit + Qdrant RAG)
# Providers: Gemini (default), Groq
# Features: custom avatars, period discipline (default 2024), LangChain-style query refine (optional),
#           version-agnostic Qdrant query, provider quotas (Gemini/Groq=8) unless user enters override key.

import os, re, json, time, uuid, traceback
from collections.abc import Mapping
from typing import List, Dict, Any, Tuple, Optional
import streamlit as st
import httpx
from dotenv import load_dotenv, find_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, PayloadSchemaType

# Optional LangChain prompt templating (graceful fallback if not installed)
try:
    from langchain_core.prompts import ChatPromptTemplate
except Exception:
    ChatPromptTemplate = None

# ==================== Secrets / Config ====================
load_dotenv(find_dotenv(filename="secrets.env", usecwd=True) or "", override=False)

def _secret(k: str, default: Optional[str] = None) -> str:
    return st.secrets.get(k, os.getenv(k, default))

# Qdrant / Embeddings (Gemini embeddings for RAG)
GEMINI_API_KEY     = _secret("GEMINI_API_KEY", "")  # hidden; only used if present
GEMINI_EMBED_MODEL = _secret("GEMINI_EMBED_MODEL", "gemini-embedding-001")
QDRANT_URL         = _secret("QDRANT_URL")
QDRANT_API_KEY     = _secret("QDRANT_API_KEY")
QDRANT_COLLECTION  = _secret("QDRANT_COLLECTION", "sec_filings")
QDRANT_VECTOR_NAME = _secret("QDRANT_VECTOR_NAME", "") or None

# Chat providers (defaults + optional secrets)
# Default provider = Gemini (best free option)
DEFAULT_PROVIDER   = "Gemini"
DEFAULT_GEM_MODEL  = _secret("GEMINI_MODEL", "gemini-1.5-flash")  # free tier
DEFAULT_GROQ_MODEL = _secret("GROQ_MODEL", "llama-3.1-8b-instant")
DEFAULT_MODEL = DEFAULT_GEM_MODEL

GROQ_API_KEY       = _secret("GROQ_API_KEY", "")    # hidden; only used if present

# Custom avatars (path/URL/emoji). You can also set BOT_AVATAR/USER_AVATAR in secrets/env.
BOT_AVATAR  = _secret("BOT_AVATAR",  os.getenv("BOT_AVATAR",  "bot.png"))
USER_AVATAR = _secret("USER_AVATAR", os.getenv("USER_AVATAR", "user.png"))
BOT_AVATAR  = BOT_AVATAR  if (BOT_AVATAR  and os.path.exists(BOT_AVATAR))  else "🧠"
USER_AVATAR = USER_AVATAR if (USER_AVATAR and os.path.exists(USER_AVATAR)) else "🧑‍💼"

SYSTEM_PROMPT = (
    "You are an expert financial analyst chatbot. Provide precise, data-driven answers based on the provided context.\n"
    "• Cite context with bracketed IDs like [1], [2] immediately after facts/numbers sourced from retrieved passages.\n"
    "• Prefer bullet points and tables where helpful.\n"
    "If context is insufficient, say exactly: \"The provided documents do not contain sufficient information to answer this question.\" "
    "Then answer using general knowledge (no citations).\n"
    "Period discipline: If a reporting period is specified by the system message, use it; if missing in the user input, assume the system’s default."
)

# ==================== UI Theme ====================
st.set_page_config(page_title="SEC Filings Chatbot", page_icon="🗠", layout="wide")
GOLD = "#D4AF37"
st.markdown(f"""
<style>
:root {{ --gold: {GOLD}; }}
html, body, [data-testid="stAppViewContainer"] {{ background:#0b0b0c; color:#e9e9e9; }}
h1,h2,h3,h4 {{ color:var(--gold); }}
.block-container {{ padding-top: 1.2rem; }}
.stButton > button, .stDownloadButton > button {{
  background:linear-gradient(135deg,#1a1a1b,#121213);
  border:1px solid var(--gold);
  color:#eaeaea; background-clip: padding-box;
  border-radius:10px; box-shadow:0 0 10px #000 inset;
}}
.chat-bubble-user {{ background:#191a1b; border:1px solid #2a2b2d; border-left:2px solid var(--gold);
  padding:10px; border-radius:12px; }}
.chat-bubble-assist {{ background:#141516; border:1px solid #2a2b2d; border-left:2px solid #8e6b12;
  padding:10px; border-radius:12px; }}

/* Gold ring on chat avatars */
[data-testid="stChatMessageAvatar"] img {{ border: 2px solid var(--gold); border-radius: 50%; }}

/* Gold creator box pinned near the bottom of the sidebar */
[data-testid="stSidebar"] .creator-box {{
  background: var(--gold); color: #0b0b0c; border: 1px solid #b68c1d;
  border-radius: 10px; padding: 10px 12px; margin-top: 12px; text-align: center; font-weight: 600;
  position: sticky; bottom: 0;
}}
[data-testid="stSidebar"] .creator-box a {{ color: #0b0b0c; text-decoration: underline; }}
</style>
""", unsafe_allow_html=True)

st.header("🗠 SEC Filings Chatbot")

# ==================== State ====================
def ensure_state():
    ss = st.session_state
    ss.setdefault("chats", [{
        "id": str(uuid.uuid4()),
        "title": "Untitled chat",
        "messages": [{"role":"system","content":SYSTEM_PROMPT}]
    }])
    ss.setdefault("active_chat_id", ss["chats"][0]["id"])
    ss.setdefault("rag_enabled", True)
    ss.setdefault("temperature", 0.3)
    ss.setdefault("top_k", 5)
    ss.setdefault("threshold", 0.0)

    # Provider + model defaults
    ss.setdefault("provider", DEFAULT_PROVIDER)  # "Gemini" | "Groq"
    ss.setdefault("model", DEFAULT_MODEL)

    # Hidden override keys (not shown unless user types)
    ss.setdefault("gemini_key_override", "")
    ss.setdefault("groq_key_override", "")

    # Per-provider question counters (for soft quotas without user-entered keys)
    ss.setdefault("q_counts", {"Groq": 0, "Gemini": 0})

    ss.setdefault("last_error", "")
    ss.setdefault("debug", False)
ensure_state()

def active_chat() -> Dict[str, Any]:
    cid = st.session_state["active_chat_id"]
    for c in st.session_state["chats"]:
        if c["id"] == cid: return c
    st.session_state["active_chat_id"] = st.session_state["chats"][0]["id"]
    return st.session_state["chats"][0]

# ==================== Clients (Qdrant only; HTTP calls per provider) ====================
@st.cache_resource
def get_qdrant_client() -> QdrantClient:
    q_timeout = float(os.getenv("QDRANT_TIMEOUT", "180"))
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=q_timeout, prefer_grpc=True)

qc = get_qdrant_client()

def ensure_payload_indexes():
    for field in ("ticker", "form", "period"):
        try:
            qc.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
                wait=True
            )
        except Exception:
            pass
ensure_payload_indexes()

# ==================== Utilities ====================
def retry_call(fn, *a, **k):
    for i in range(4):
        try: return fn(*a, **k)
        except Exception:
            if i == 3: raise
            time.sleep(min(2.0*(i+1), 6.0))


def _vector_kwargs() -> Dict[str, Any]:
    return {"vector_name": QDRANT_VECTOR_NAME} if QDRANT_VECTOR_NAME else {}

def safe_err(e: BaseException) -> str:
    try: return "".join(traceback.format_exception_only(type(e), e)).strip()
    except Exception:
        try: return repr(e)
        except Exception: return "<unprintable exception>"

@st.cache_data(ttl=120, show_spinner=False)
def collection_vector_size() -> Optional[int]:
    """Return the dimensionality of the configured vector set in Qdrant."""
    try:
        info = qc.get_collection(QDRANT_COLLECTION)
        vectors = getattr(info.config.params, "vectors", None)
        if vectors is None:
            return None
        # Single unnamed vector configuration
        if hasattr(vectors, "size"):
            return getattr(vectors, "size", None)
        # Named vectors configuration
        if isinstance(vectors, Mapping):
            if QDRANT_VECTOR_NAME and QDRANT_VECTOR_NAME in vectors:
                target = vectors[QDRANT_VECTOR_NAME]
                size = getattr(target, "size", None) or getattr(target, "dimension", None)
                if size:
                    return size
                if isinstance(target, dict):
                    return target.get("size") or target.get("dimension")
            # fallback to first defined vector
            for target in vectors.values():
                if target is None:
                    continue
                size = getattr(target, "size", None) or getattr(target, "dimension", None)
                if size:
                    return size
                if isinstance(target, dict):
                    size = target.get("size") or target.get("dimension")
                    if size:
                        return size
        # Some client versions expose named vectors via attribute instead of mapping
        if hasattr(vectors, "__dict__") and isinstance(vectors.__dict__, dict):
            maybe = vectors.__dict__
            if QDRANT_VECTOR_NAME and QDRANT_VECTOR_NAME in maybe:
                target = maybe[QDRANT_VECTOR_NAME]
                size = getattr(target, "size", None) or getattr(target, "dimension", None)
                if size:
                    return size
                if isinstance(target, dict):
                    return target.get("size") or target.get("dimension")
            for target in maybe.values():
                if target is None:
                    continue
                size = getattr(target, "size", None) or getattr(target, "dimension", None)
                if size:
                    return size
        return None
    except Exception:
        return None

@st.cache_data(ttl=300)
def discover_facets(page_size: int = 500) -> Tuple[List[str], List[str]]:
    tickers, forms = set(), set()
    offset = None
    while True:
        pts, offset = qc.scroll(QDRANT_COLLECTION, with_payload=True, limit=page_size, offset=offset)
        if not pts: break
        for p in pts:
            pl = p.payload or {}
            t = pl.get("ticker"); f = pl.get("form")
            if t: tickers.add(str(t))
            if f: forms.add(str(f))
        if offset is None: break
    return sorted(tickers), sorted(forms)

try:
    TICKERS, FORMS = discover_facets()
except Exception:
    TICKERS, FORMS = [], []
    st.toast("Could not load tickers/forms from Qdrant; RAG still works without selectors.", icon="⚠️")

# Restrict tickers to allowlist & default
ALLOWED_TICKERS = ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","AMD","TSLA","AVGO","INTC"]
TICKERS = [t for t in TICKERS if t in ALLOWED_TICKERS]

def qdrant_count_safe(collection: str, flt) -> int:
    try:
        return qc.count(collection, exact=True, filter=flt).count
    except Exception as e1:
        m = str(e1)
        if "Unknown arguments" in m and "filter" in m:
            try:
                return qc.count(collection, exact=True, count_filter=flt).count
            except Exception as e2:
                m2 = str(e2)
                if "Unknown arguments" in m2 and "count_filter" in m2:
                    return qc.count(collection, exact=True, query_filter=flt).count
                raise
        if isinstance(e1, TypeError):
            try: return qc.count(collection, exact=True, count_filter=flt).count
            except Exception: return qc.count(collection, exact=True, query_filter=flt).count
        raise

# ==================== Sidebar ====================
with st.sidebar:
    st.subheader("Session")
    if st.button("➕ New Chat", use_container_width=True):
        nid = str(uuid.uuid4())
        st.session_state["chats"].insert(0, {
            "id": nid, "title": "Untitled chat", "messages":[{"role":"system","content":SYSTEM_PROMPT}]
        })
        st.session_state["active_chat_id"] = nid
        # reset per-provider counters on fresh chat
        st.session_state["q_counts"] = {"Groq": 0, "Gemini": 0}

    st.write("History")
    for c in st.session_state["chats"]:
        label = (c["title"][:32] + "…") if len(c["title"]) > 33 else c["title"]
        if st.button(label, key=f"chat_{c['id']}", use_container_width=True):
            st.session_state["active_chat_id"] = c["id"]

    st.divider()
    st.subheader("RAG")
    st.session_state["rag_enabled"] = st.toggle("Enable RAG (Qdrant)", value=st.session_state["rag_enabled"])

    with st.expander("Advanced Settings", expanded=False):
        # Provider + models (default Gemini)
        provider_options = ["Gemini", "Groq"]
        st.session_state["provider"] = st.selectbox("Provider", provider_options, index=provider_options.index(DEFAULT_PROVIDER))
        if st.session_state["provider"] == "Gemini":
            models = [DEFAULT_GEM_MODEL, "gemini-1.5-flash-8b", "gemini-1.5-pro"]
        else:
            models = [DEFAULT_GROQ_MODEL, "llama-3.1-70b-versatile", "mixtral-8x7b-32768"]
        st.session_state["model"] = st.selectbox("Model", options=list(dict.fromkeys(models)), index=0)

        st.session_state["top_k"] = st.slider("Top K", 3, 10, st.session_state["top_k"], 1)
        st.session_state["threshold"] = st.slider("Min score", 0.0, 1.0, st.session_state["threshold"], 0.05)
        st.session_state["temperature"] = st.slider("Temperature", 0.0, 1.0, st.session_state["temperature"], 0.05)

        # Optional override keys (hidden)
        st.session_state["gemini_key_override"] = st.text_input("Gemini API Key (override)", type="password", value=st.session_state.get("gemini_key_override",""))
        st.session_state["groq_key_override"]   = st.text_input("Groq API Key (override)",   type="password", value=st.session_state.get("groq_key_override",""))

    st.divider()
    cur = active_chat()
    st.download_button("Export current chat (.json)",
        data=json.dumps(cur, ensure_ascii=False, indent=2),
        file_name=f"chat_{cur['id']}.json", mime="application/json",
        use_container_width=True)

    with st.expander("🔎 RAG Diagnostics", expanded=False):
        if st.button("Run checks", use_container_width=True):
            try:
                size = collection_vector_size()
                st.write("Collection dim:", size)
                st.write("Vector name:", QDRANT_VECTOR_NAME or "<default>")
                # Embedding dim ping (Gemini embeddings)
                gem_key = (st.session_state.get("gemini_key_override") or GEMINI_API_KEY or "").strip()
                if gem_key:
                    embed_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_EMBED_MODEL}:embedContent?key={gem_key}"
                    payload = {"content": {"parts": [{"text": "diagnostic ping"}]}}
                    r = httpx.post(embed_url, json=payload, timeout=httpx.Timeout(20.0, connect=10.0))
                    r.raise_for_status()
                    embedding = r.json().get("embedding", {})
                    vector = embedding.get("values") or embedding.get("value") or []
                    emb_dim = len(vector); st.write("Embed dim:", emb_dim)
                    if size and emb_dim != size:
                        st.error(f"Dimension mismatch. Collection={size}, Embed={emb_dim} ({GEMINI_EMBED_MODEL})")
                else:
                    st.warning("Gemini key missing; embeddings (RAG) will be disabled.", icon="⚠️")
                cnt_all = qc.count(QDRANT_COLLECTION, exact=True).count; st.write("Points count:", cnt_all)
                sel_tick = st.session_state.get("sel_ticker"); sel_form = st.session_state.get("sel_form")
                if sel_tick and sel_form:
                    flt = Filter(must=[FieldCondition(key="ticker", match=MatchValue(value=sel_tick)),
                                       FieldCondition(key="form",   match=MatchValue(value=sel_form))])
                    cnt_f = qdrant_count_safe(QDRANT_COLLECTION, flt)
                    st.write(f"Filtered count ({sel_tick}, {sel_form}):", cnt_f)
            except Exception as e:
                st.error(f"Diag error: {safe_err(e)}")

    # Creator box
    st.markdown(
        """
        <div class="creator-box">
          © Created by Palash Dubey<br/>
          (<a href="https://github.com/Palash-creator" target="_blank">GitHub</a> |
           <a href="https://www.linkedin.com/in/palash-dubey/" target="_blank">LinkedIn</a>)
        </div>
        """,
        unsafe_allow_html=True
    )

# ==================== Top selectors ====================
sc1, sc2, sc3 = st.columns([1.2, 1.2, 2.6])
with sc1:
    DEFAULT_TICKER = "AAPL"
    if "sel_ticker" not in st.session_state:
        st.session_state["sel_ticker"] = (DEFAULT_TICKER if DEFAULT_TICKER in TICKERS else (TICKERS[0] if TICKERS else None))
    sel_ticker = st.selectbox("Ticker", TICKERS, key="sel_ticker", placeholder="Select ticker")
with sc2:
    sel_form = st.selectbox("Form", FORMS, index=0 if FORMS else None, key="sel_form", placeholder="Select form")
with sc3:
    st.caption("Select a ticker & form to enable retrieval. Use sidebar to tune RAG.")

st.divider()

# ==================== Transcript ====================
chat = active_chat()
for m in chat["messages"][1:]:  # skip system
    avatar = USER_AVATAR if m["role"] == "user" else BOT_AVATAR
    with st.chat_message(m["role"], avatar=avatar):
        klass = "chat-bubble-user" if m["role"] == "user" else "chat-bubble-assist"
        st.markdown(f"<div class='{klass}'>{m['content']}</div>", unsafe_allow_html=True)

# ==================== Core RAG helpers ====================
@st.cache_data(ttl=90, show_spinner=False)
def embed_query(text: str) -> List[float]:
    # Embeddings always via Gemini (only if key present)
    key = (st.session_state.get("gemini_key_override") or GEMINI_API_KEY or "").strip()
    if not key:
        raise RuntimeError("Gemini API key is required for embeddings.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_EMBED_MODEL}:embedContent?key={key}"
    payload = {"content": {"parts": [{"text": text}]}}
    r = retry_call(httpx.post, url, json=payload, timeout=httpx.Timeout(20.0, connect=10.0))
    r.raise_for_status()
    data = r.json()
    embedding = data.get("embedding", {})
    vector = embedding.get("values") or embedding.get("value") or []
    if not vector:
        raise RuntimeError("Gemini embedding response missing vector values.")
    return vector

def _q_query(qvec, flt, k, thr):
    vec_kwargs = _vector_kwargs()
    try:
        return qc.query_points(collection_name=QDRANT_COLLECTION, query=qvec, limit=k, with_payload=True, filter=flt, score_threshold=(thr or None), **vec_kwargs)
    except Exception as e:
        msg = str(e)
        if "Unknown arguments" in msg and "filter" in msg:
            try:
                return qc.query_points(collection_name=QDRANT_COLLECTION, query=qvec, limit=k, with_payload=True, query_filter=flt, score_threshold=(thr or None), **vec_kwargs)
            except Exception as e2:
                if "Unknown arguments" in str(e2) and "score_threshold" in str(e2):
                    return qc.query_points(collection_name=QDRANT_COLLECTION, query=qvec, limit=k, with_payload=True, query_filter=flt, **vec_kwargs)
        if "Unknown arguments" in msg and "score_threshold" in msg:
            try:
                return qc.query_points(collection_name=QDRANT_COLLECTION, query=qvec, limit=k, with_payload=True, filter=flt, **vec_kwargs)
            except Exception:
                pass
    try:
        return qc.search(collection_name=QDRANT_COLLECTION, query_vector=qvec, limit=k, with_payload=True, query_filter=flt, **vec_kwargs)
    except Exception:
        if vec_kwargs:
            return qc.search(collection_name=QDRANT_COLLECTION, query_vector=qvec, limit=k, with_payload=True, query_filter=flt)
        raise

def _iter_hits(hits):
    pts = getattr(hits, "points", hits)
    return pts or []

# === Reporting period helpers ===
DEFAULT_PERIOD = "2024"
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

def detect_period(text: str) -> str:
    matches = [m.group(0) for m in YEAR_RE.finditer(text)]
    return matches[-1] if matches else DEFAULT_PERIOD

# === Query refinement (LangChain if available; safe fallback otherwise) ===
def refine_prompt(user_text: str, ticker: Optional[str], form: Optional[str], period: str) -> str:
    system = (
        "Rewrite the user's question as a concise query specific to the given filing scope.\n"
        f"Scope: ticker={ticker or 'UNKNOWN'}, form={form or 'UNKNOWN'}, period={period}.\n"
        "Keep only relevant constraints. Output a single sentence, no extra commentary."
    )
    if ChatPromptTemplate:
        tmpl = ChatPromptTemplate.from_messages([("system", system), ("user", "{question}")])
        provider = st.session_state.get("provider", DEFAULT_PROVIDER)
        mdl = DEFAULT_GROQ_MODEL if provider == "Groq" else DEFAULT_GEM_MODEL
        rewritten = _llm_call(provider, mdl, tmpl.format_messages(question=user_text), temperature=0.0)
        return rewritten.strip() if rewritten else f"{user_text} (scope: {ticker} {form} {period})"
    return f"{user_text}\n\n(Answer strictly from {ticker or 'the selected ticker'} {form or 'the selected form'} for reporting period {period}.)"

# === Provider-agnostic chat ===
def _llm_call(provider: str, model: str, messages: List[Dict[str, str]] | Any, temperature: float = 0.2) -> str:
    # Normalize messages to chat-completion-style list
    norm_msgs: List[Dict[str,str]] = []
    if isinstance(messages, list) and messages and isinstance(messages[0], dict):
        norm_msgs = messages
    else:
        try:
            for m in messages:
                role = getattr(m, "type", None) or getattr(m, "role", None) or "user"
                content = getattr(m, "content", "")
                if role == "human": role = "user"
                norm_msgs.append({"role": role, "content": content})
        except Exception:
            norm_msgs = [{"role": "user", "content": str(messages)}]

    if provider == "Gemini":
        key = (st.session_state.get("gemini_key_override") or GEMINI_API_KEY or "").strip()
        if not key:
            raise httpx.HTTPError("Gemini key missing. Enter one in Advanced Settings.")
        # Build Gemini payload
        sys_texts = [m["content"] for m in norm_msgs if m["role"] == "system"]
        user_assistant_msgs = [m for m in norm_msgs if m["role"] in ("user","assistant")]
        contents = []
        for m in user_assistant_msgs:
            contents.append({"role": "user" if m["role"]=="user" else "model",
                             "parts":[{"text": m["content"]}]})
        body = {"contents": contents, "generationConfig": {"temperature": temperature}}
        if sys_texts:
            body["systemInstruction"] = {"role":"system","parts":[{"text":"\n\n".join(sys_texts)}]}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        r = retry_call(httpx.post, url, json=body, timeout=httpx.Timeout(30.0, connect=10.0))
        r.raise_for_status()
        data = r.json()
        # Extract text
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            raise httpx.HTTPError("Gemini response parsing failed.")
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    if provider == "Groq":
        key = (st.session_state.get("groq_key_override") or GROQ_API_KEY or "").strip()
        if not key:
            raise httpx.HTTPError("Groq key missing. Enter one in Advanced Settings.")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        body = {"model": model, "temperature": temperature, "messages": norm_msgs}
        r = retry_call(httpx.post, "https://api.groq.com/openai/v1/chat/completions",
                       json=body, headers=headers, timeout=httpx.Timeout(30.0, connect=10.0))
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    raise ValueError(f"Unsupported provider: {provider}")

def retrieve(qvec: List[float], ticker: str, form: str, period: str, k: int, thr: float) -> List[Dict[str, Any]]:
    coll_dim = collection_vector_size()
    if coll_dim and coll_dim != len(qvec):
        vname = QDRANT_VECTOR_NAME or "default"
        raise RuntimeError(
            f"Vector size mismatch: collection={coll_dim}, embedding={len(qvec)} for {GEMINI_EMBED_MODEL} (vector {vname})."
        )
    must = [FieldCondition(key="ticker", match=MatchValue(value=ticker)),
            FieldCondition(key="form",   match=MatchValue(value=form))]
    if period:
        must.append(FieldCondition(key="period", match=MatchValue(value=str(period))))
    flt = Filter(must=must)
    try:
        hits = _q_query(qvec, flt, k, thr)
    except Exception as e:
        if "Index required but not found" in str(e):
            ensure_payload_indexes(); time.sleep(0.3)
            hits = _q_query(qvec, flt, k, thr)
        else:
            raise
    pts = list(_iter_hits(hits))
    if not pts:
        try:
            uf = _q_query(qvec, None, k, thr)
            pts = list(_iter_hits(uf))
            if pts:
                st.toast("No matches for current filters; showing unfiltered results.", icon="ℹ️")
        except Exception:
            pass
    docs: List[Dict[str, Any]] = []
    for i, h in enumerate(pts, 1):
        sc = float(getattr(h, "score", 0.0))
        if thr and sc < thr: continue
        pl = h.payload or {}
        docs.append({
            "id": i, "text": pl.get("text",""), "score": sc,
            "meta": {"source_path": pl.get("source_path"),
                     "chunk_idx": pl.get("chunk_idx"),
                     "ticker": pl.get("ticker"),
                     "form": pl.get("form"),
                     "period": pl.get("period")}
        })
    return docs

def build_context(docs: List[Dict[str, Any]], max_chunks: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
    take = docs[:max_chunks]
    ctx = "\n\n".join(f"[{d['id']}] {d['text']}" for d in take if d.get("text"))
    return ctx, take

def call_llm(provider: str, model: str, user_text: str, context: Optional[str], temperature: float, period: str) -> str:
    msgs = [{"role":"system","content":SYSTEM_PROMPT}]
    if period:
        msgs.append({"role":"system","content":f"Reporting period to use: {period}. "
                                               f"If the user does not specify a period, assume this for all figures/statements."})
    if context:
        msgs.append({"role":"system","content":"Context:\n"+context})
    msgs.append({"role":"user","content":user_text})
    try:
        return _llm_call(provider, model, msgs, temperature=temperature)
    except httpx.HTTPStatusError:
        # Fallbacks
        if provider == "Groq":
            if model != DEFAULT_GROQ_MODEL:
                return _llm_call("Groq", DEFAULT_GROQ_MODEL, msgs, temperature=temperature)
        else:  # Gemini
            if model != DEFAULT_GEM_MODEL:
                return _llm_call("Gemini", DEFAULT_GEM_MODEL, msgs, temperature=temperature)
        raise

# ==================== Quota checks ====================
def provider_key_present(provider: str) -> bool:
    if provider == "Gemini":
        return bool((st.session_state.get("gemini_key_override") or GEMINI_API_KEY).strip())
    if provider == "Groq":
        return bool((st.session_state.get("groq_key_override") or GROQ_API_KEY).strip())
    return False

def quota_limit(provider: str) -> int:
    return 8

def check_quota_and_maybe_block(provider: str) -> bool:
    """
    Returns True if we should BLOCK (i.e., stop answering) due to quota without override key.
    Policy: Gemini and Groq stop after 8 questions. If the user types an override key for that
    provider, the quota is lifted.
    """
    counts = st.session_state.get("q_counts", {"Groq":0,"Gemini":0})
    count = counts.get(provider, 0)
    limit = quota_limit(provider)
    if count >= limit and not st.session_state.get(f"{provider.lower()}_key_override", ""):
        msg = (f"You've reached the free quota for **{provider}** "
               f"({limit} questions). Enter a {provider} API key in **Advanced Settings**, "
               f"or switch Provider (Gemini/Groq).")
        st.warning(msg, icon="⚠️")
        return True
    return False

def incr_quota(provider: str):
    counts = st.session_state.get("q_counts", {"Groq":0,"Gemini":0})
    counts[provider] = counts.get(provider, 0) + 1
    st.session_state["q_counts"] = counts

# ==================== Bottom chat input (continuous echo + avatars + refine + period default) ====================
prompt = st.chat_input("Select Ticker & Form, then ask about SEC filings (default period 2024).")

if prompt:
    provider = st.session_state.get("provider", DEFAULT_PROVIDER)
    model = st.session_state.get("model", DEFAULT_GEM_MODEL)

    # Show the *current* user message immediately
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(f"<div class='chat-bubble-user'>{prompt}</div>", unsafe_allow_html=True)

    # Quota check BEFORE doing any work
    if check_quota_and_maybe_block(provider):
        # do not proceed; do not increment counters
        pass
    else:
        # Persist to history and increment provider counter
        chat["messages"].append({"role": "user", "content": prompt})
        incr_quota(provider)

        if chat["title"] == "Untitled chat":
            chat["title"] = (prompt.strip().split("\n")[0])[:40] or "Untitled chat"

        # Derive/assume reporting period
        period = detect_period(prompt)

        # Optional query refinement to align with ticker/form/period
        try:
            refined = refine_prompt(prompt, st.session_state.get("sel_ticker"), st.session_state.get("sel_form"), period)
        except Exception:
            refined = f"{prompt}\n\n(Assume reporting period {period})"

        # RAG toggle & selections
        use_rag = bool(
            st.session_state["rag_enabled"]
            and st.session_state.get("sel_ticker")
            and st.session_state.get("sel_form")
        )
        docs, ctx, qvec = [], None, None

        if use_rag:
            try:
                with st.spinner("Embedding refined query…"):
                    qvec = embed_query(refined)
            except Exception as e:
                st.session_state["last_error"] = f"Embed error: {safe_err(e)}"
                st.toast("Embedding failed or Gemini key missing. Falling back to non-RAG.", icon="⚠️")
                use_rag = False

        if use_rag and qvec is not None:
            try:
                with st.spinner("Retrieving…"):
                    docs = retrieve(
                        qvec,
                        st.session_state["sel_ticker"],
                        st.session_state["sel_form"],
                        period,
                        st.session_state["top_k"],
                        st.session_state["threshold"],
                    )
                    ctx, _ = build_context(docs)
            except Exception as e:
                st.session_state["last_error"] = f"Retrieval error: {safe_err(e)}"
                st.toast("RAG retrieval failed. Using non-RAG.", icon="⚠️")
                use_rag = False
                ctx = None

        with st.spinner("Generating…"):
            try:
                reply = call_llm(provider, model, prompt, ctx, st.session_state["temperature"], period=period)
            except Exception as e:
                st.session_state["last_error"] = f"Provider error: {safe_err(e)}"
                reply = "Sorry—there was an issue generating a response."

        # Persist and render assistant reply
        chat["messages"].append({"role": "assistant", "content": reply})
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.markdown(f"<div class='chat-bubble-assist'>{reply}</div>", unsafe_allow_html=True)

        if use_rag and docs:
            with st.expander("Sources"):
                for d in docs:
                    src = d["meta"].get("source_path", "unknown")
                    st.markdown(f"**[{d['id']}]** {src} — score={d['score']:.3f}")
                    st.caption((d["text"][:400] + "…") if len(d["text"]) > 400 else d["text"])

# ==================== Debug (optional) ====================
if st.session_state.get("debug") and st.session_state.get("last_error"):
    st.info(st.session_state["last_error"])
