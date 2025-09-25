# app.py — SEC Filings Chatbot (Streamlit + Qdrant RAG) with custom avatars
import os, json, time, uuid, traceback
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
import httpx
from dotenv import load_dotenv, find_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, PayloadSchemaType
)

# ==================== Secrets / Config ====================
load_dotenv(find_dotenv(filename="secrets.env", usecwd=True) or "", override=False)

def _secret(k: str, default: Optional[str] = None) -> str:
    return st.secrets.get(k, os.getenv(k, default))

OPENAI_API_KEY     = _secret("OPENAI_API_KEY")
OPENAI_MODEL       = _secret("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = _secret("OPENAI_EMBED_MODEL", "text-embedding-3-large")  # 3072-d
QDRANT_URL         = _secret("QDRANT_URL")
QDRANT_API_KEY     = _secret("QDRANT_API_KEY")
QDRANT_COLLECTION  = _secret("QDRANT_COLLECTION", "sec_filings")

# Custom avatars (file path, URL, or emoji). You can also set BOT_AVATAR/USER_AVATAR in env or st.secrets.
BOT_AVATAR  = _secret("BOT_AVATAR",  os.getenv("BOT_AVATAR",  "assets/bot.png"))
USER_AVATAR = _secret("USER_AVATAR", os.getenv("USER_AVATAR", "assets/user.png"))
BOT_AVATAR  = BOT_AVATAR  if (BOT_AVATAR  and os.path.exists(BOT_AVATAR))  else "🤖"
USER_AVATAR = USER_AVATAR if (USER_AVATAR and os.path.exists(USER_AVATAR)) else "🧑‍💻"

SYSTEM_PROMPT = (
    """You are an expert financial analyst chatbot. Your primary purpose is to provide precise, data-driven answers to financial questions by analyzing the context provided from source documents. Maintain the highest industry standards in your reporting.
    
**Core Instructions**:
Analyze and Report: Your main goal is to extract and report information found within the provided context. Base your answers strictly on this data.

**Formatting and Citation**:
Structure your answers for maximum clarity. Use bullet points and markdown tables whenever possible.
You MUST cite your sources. At the end of any sentence or data point drawn from a source, add its corresponding bracketed ID, like [1] or [3].

**Handling Insufficient Context**:
If the provided context does not contain the information needed to answer the question, you must explicitly state: "The provided documents do not contain sufficient information to answer this question."
After stating this, you should then try to answer the question using your general knowledge as an expert financial analyst. Do not use citations for this general knowledge.
Be thorough and professional. Your responses should be accurate, clear, and helpful."""
)

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in secrets."); st.stop()

# ==================== UI Theme ====================
st.set_page_config(page_title="SEC Filings Chatbot", page_icon="📄", layout="wide")
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
[data-testid="stChatMessageAvatar"] img {{
  border: 2px solid var(--gold);
  border-radius: 50%;
}}

/* Gold creator box pinned near the bottom of the sidebar */
[data-testid="stSidebar"] .creator-box {{
  background: var(--gold);
  color: #0b0b0c;
  border: 1px solid #b68c1d;
  border-radius: 10px;
  padding: 10px 12px;
  margin-top: 12px;
  text-align: center;
  font-weight: 600;
  position: sticky;
  bottom: 0;
}}
[data-testid="stSidebar"] .creator-box a {{
  color: #0b0b0c;
  text-decoration: underline;
}}
</style>
""", unsafe_allow_html=True)

st.header("SEC Filings Chatbot")

# ==================== Clients (cached) ====================
@st.cache_resource
def get_clients() -> Tuple[httpx.Client, QdrantClient]:
    http = httpx.Client(
        timeout=httpx.Timeout(20.0, connect=10.0),
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    )
    q_timeout = float(os.getenv("QDRANT_TIMEOUT", "180"))
    qc = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=q_timeout,
        prefer_grpc=True,
    )
    return http, qc

http, qc = get_clients()

def ensure_payload_indexes():
    for field in ("ticker", "form"):
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
    ss.setdefault("model", OPENAI_MODEL)
    ss.setdefault("last_error", "")
    ss.setdefault("debug", False)
ensure_state()

def active_chat() -> Dict[str, Any]:
    cid = st.session_state["active_chat_id"]
    for c in st.session_state["chats"]:
        if c["id"] == cid: return c
    st.session_state["active_chat_id"] = st.session_state["chats"][0]["id"]
    return st.session_state["chats"][0]

# ==================== Utilities ====================
def retry_call(fn, *a, **k):
    for i in range(4):
        try: return fn(*a, **k)
        except Exception:
            if i == 3: raise
            time.sleep(min(2.0*(i+1), 6.0))

def safe_err(e: BaseException) -> str:
    try:
        return "".join(traceback.format_exception_only(type(e), e)).strip()
    except Exception:
        try: return repr(e)
        except Exception: return "<unprintable exception>"

@st.cache_data(ttl=120, show_spinner=False)
def collection_vector_size() -> Optional[int]:
    try:
        info = qc.get_collection(QDRANT_COLLECTION)
        vec = info.config.params.vectors
        return getattr(vec, "size", None)
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

# Restrict tickers to allowlist & default to AAPL
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
            try:
                return qc.count(collection, exact=True, count_filter=flt).count
            except Exception:
                return qc.count(collection, exact=True, query_filter=flt).count
        raise

# ==================== Sidebar ====================
with st.sidebar:
    st.subheader("Session")
    if st.button("➕ New Chat", use_container_width=True):
        nid = str(uuid.uuid4())
        st.session_state["chats"].insert(0, {
            "id": nid, "title": "Untitled chat",
            "messages":[{"role":"system","content":SYSTEM_PROMPT}]
        })
        st.session_state["active_chat_id"] = nid

    st.write("History")
    for c in st.session_state["chats"]:
        label = (c["title"][:32] + "…") if len(c["title"]) > 33 else c["title"]
        if st.button(label, key=f"chat_{c['id']}", use_container_width=True):
            st.session_state["active_chat_id"] = c["id"]

    st.divider()
    st.subheader("RAG")
    st.session_state["rag_enabled"] = st.toggle("Enable RAG (Qdrant)", value=st.session_state["rag_enabled"])

    with st.expander("Advanced Settings", expanded=False):
        st.session_state["top_k"] = st.slider("Top K", 3, 10, st.session_state["top_k"], 1)
        st.session_state["threshold"] = st.slider("Min score", 0.0, 1.0, st.session_state["threshold"], 0.05)
        st.session_state["temperature"] = st.slider("Temperature", 0.0, 1.0, st.session_state["temperature"], 0.05)
        valid_models = list(dict.fromkeys([OPENAI_MODEL, "gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"]))
        st.session_state["model"] = st.selectbox("Model", options=valid_models, index=0)

    st.divider()
    cur = active_chat()
    st.download_button("Export current chat (.json)",
        data=json.dumps(cur, ensure_ascii=False, indent=2),
        file_name=f"chat_{cur['id']}.json", mime="application/json",
        use_container_width=True)

    with st.expander("🔎 RAG Diagnostics", expanded=False):
        if st.button("Run checks", use_container_width=True):
            try:
                size = collection_vector_size(); st.write("Collection dim:", size)
                payload = {"model": OPENAI_EMBED_MODEL, "input": "diagnostic ping"}
                r = http.post("https://api.openai.com/v1/embeddings", json=payload); r.raise_for_status()
                emb_dim = len(r.json()["data"][0]["embedding"]); st.write("Embed dim:", emb_dim)
                if size and emb_dim != size:
                    st.error(f"Dimension mismatch. Collection={size}, Embed={emb_dim} ({OPENAI_EMBED_MODEL})")
                cnt_all = qc.count(QDRANT_COLLECTION, exact=True).count; st.write("Points count:", cnt_all)
                sel_tick = st.session_state.get("sel_ticker"); sel_form = st.session_state.get("sel_form")
                if sel_tick and sel_form:
                    flt = Filter(must=[
                        FieldCondition(key="ticker", match=MatchValue(value=sel_tick)),
                        FieldCondition(key="form",   match=MatchValue(value=sel_form)),
                    ])
                    cnt_f = qdrant_count_safe(QDRANT_COLLECTION, flt)
                    st.write(f"Filtered count ({sel_tick}, {sel_form}):", cnt_f)
            except Exception as e:
                st.error(f"Diag error: {safe_err(e)}")

    # Creator box (gold) pinned near sidebar bottom
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
        st.session_state["sel_ticker"] = (
            DEFAULT_TICKER if DEFAULT_TICKER in TICKERS
            else (TICKERS[0] if TICKERS else None)
        )
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
    payload = {"model": OPENAI_EMBED_MODEL, "input": text}
    r = retry_call(http.post, "https://api.openai.com/v1/embeddings", json=payload)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]

def _q_query(qvec, flt, k, thr):
    """
    Version-agnostic Qdrant query:
      1) query_points(..., filter=...)
      2) query_points(..., query_filter=...)
      3) search(..., query_filter=...)  (legacy)
    Also tolerates servers that don't accept score_threshold.
    """
    try:
        return qc.query_points(
            collection_name=QDRANT_COLLECTION,
            query=qvec,
            limit=k,
            with_payload=True,
            filter=flt,
            score_threshold=(thr or None),
        )
    except Exception as e:
        msg = str(e)
        if "Unknown arguments" in msg and "filter" in msg:
            try:
                return qc.query_points(
                    collection_name=QDRANT_COLLECTION,
                    query=qvec,
                    limit=k,
                    with_payload=True,
                    query_filter=flt,
                    score_threshold=(thr or None),
                )
            except Exception as e2:
                msg2 = str(e2)
                if "Unknown arguments" in msg2 and "score_threshold" in msg2:
                    return qc.query_points(
                        collection_name=QDRANT_COLLECTION,
                        query=qvec,
                        limit=k,
                        with_payload=True,
                        query_filter=flt,
                    )
        if "Unknown arguments" in msg and "score_threshold" in msg:
            try:
                return qc.query_points(
                    collection_name=QDRANT_COLLECTION,
                    query=qvec,
                    limit=k,
                    with_payload=True,
                    filter=flt,
                )
            except Exception:
                pass
    return qc.search(  # legacy
        collection_name=QDRANT_COLLECTION,
        query_vector=qvec,
        limit=k,
        with_payload=True,
        query_filter=flt,
    )

def _iter_hits(hits):
    pts = getattr(hits, "points", hits)
    return pts or []

def retrieve(qvec: List[float], ticker: str, form: str, k: int, thr: float) -> List[Dict[str, Any]]:
    coll_dim = collection_vector_size()
    if coll_dim and coll_dim != len(qvec):
        raise RuntimeError(f"Vector size mismatch: collection={coll_dim}, embedding={len(qvec)} for {OPENAI_EMBED_MODEL}.")

    flt = Filter(must=[
        FieldCondition(key="ticker", match=MatchValue(value=ticker)),
        FieldCondition(key="form",   match=MatchValue(value=form)),
    ])

    try:
        hits = _q_query(qvec, flt, k, thr)
    except Exception as e:
        if "Index required but not found" in str(e):
            ensure_payload_indexes(); time.sleep(0.3)
            hits = _q_query(qvec, flt, k, thr)
        else:
            raise

    pts = list(_iter_hits(hits))
    if not pts:  # diagnostic fallback
        try:
            uf = _q_query(qvec, None, k, thr)
            pts = list(_iter_hits(uf))
            if pts:
                st.toast("No matches for current Ticker/Form; showing unfiltered results.", icon="ℹ️")
        except Exception:
            pass

    docs: List[Dict[str, Any]] = []
    for i, h in enumerate(pts, 1):
        sc = float(getattr(h, "score", 0.0))
        if thr and sc < thr:
            continue
        pl = h.payload or {}
        docs.append({
            "id": i, "text": pl.get("text",""), "score": sc,
            "meta": {"source_path": pl.get("source_path"),
                     "chunk_idx": pl.get("chunk_idx"),
                     "ticker": pl.get("ticker"),
                     "form": pl.get("form")}
        })
    return docs

def build_context(docs: List[Dict[str, Any]], max_chunks: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
    take = docs[:max_chunks]
    ctx = "\n\n".join(f"[{d['id']}] {d['text']}" for d in take if d.get("text"))
    return ctx, take

def call_llm(model: str, user_text: str, context: Optional[str], temperature: float) -> str:
    msgs = [{"role":"system","content":SYSTEM_PROMPT}]
    if context:
        msgs.append({"role":"system","content":"Context:\n"+context})
    msgs.append({"role":"user","content":user_text})

    def _invoke(m):
        r = retry_call(http.post, "https://api.openai.com/v1/chat/completions",
                       json={"model": m, "temperature": temperature, "messages": msgs})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    try:
        return _invoke(model)
    except httpx.HTTPStatusError as e:
        if e.response is not None and e.response.status_code in (400, 404) and model != OPENAI_MODEL:
            return _invoke(OPENAI_MODEL)
        raise

# ==================== Bottom chat input (continuous echo + custom avatars) ====================
prompt = st.chat_input("Select a Ticker and ask questions on SEC filings!")

if prompt:
    # Show the *current* user message immediately
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(f"<div class='chat-bubble-user'>{prompt}</div>", unsafe_allow_html=True)

    # Persist to history
    chat["messages"].append({"role": "user", "content": prompt})
    if chat["title"] == "Untitled chat":
        chat["title"] = (prompt.strip().split("\n")[0])[:40] or "Untitled chat"

    # RAG toggle & selections
    use_rag = bool(
        st.session_state["rag_enabled"]
        and st.session_state.get("sel_ticker")
        and st.session_state.get("sel_form")
    )
    docs, ctx, qvec = [], None, None

    if use_rag:
        try:
            with st.spinner("Embedding query…"):
                qvec = embed_query(prompt)
        except Exception as e:
            st.session_state["last_error"] = f"Embed error: {safe_err(e)}"
            st.toast("Embedding failed. Falling back to non-RAG.", icon="⚠️")
            use_rag = False

    if use_rag and qvec is not None:
        try:
            with st.spinner("Retrieving…"):
                docs = retrieve(
                    qvec,
                    st.session_state["sel_ticker"],
                    st.session_state["sel_form"],
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
            reply = call_llm(
                st.session_state["model"],
                prompt,
                ctx,
                st.session_state["temperature"],
            )
        except Exception as e:
            st.session_state["last_error"] = f"OpenAI error: {safe_err(e)}"
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
