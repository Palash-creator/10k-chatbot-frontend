import itertools
import json
import time
import uuid
from typing import Dict, List, Optional, Tuple

import httpx
import streamlit as st
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue


class Settings(BaseModel):
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBED_MODEL: str = "text-embedding-3-large"
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: Optional[str] = None


st.set_page_config(page_title="Gold & Black Analyst", page_icon="✨", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background:#0b0b0c;color:#f4e9c9;}
    section[data-testid="stSidebar"] {background:linear-gradient(180deg,#111112 0%,#050506 100%);border-right:1px solid #2a2a2d;}
    section[data-testid="stSidebar"] button {border-radius:18px;background:rgba(212,175,55,0.15);border:1px solid rgba(212,175,55,0.35);color:#f5ddb2;font-weight:600;}
    section[data-testid="stSidebar"] button:hover {background:rgba(212,175,55,0.3);border-color:#d4af37;}
    .chat-chip {display:inline-flex;align-items:center;gap:0.35rem;background:rgba(212,175,55,0.18);border-radius:999px;padding:0.25rem 0.9rem;font-size:0.75rem;color:#f7df9d;border:1px solid rgba(212,175,55,0.35);}
    [data-testid="stChatMessage"] {background:rgba(255,255,255,0.04);border:1px solid rgba(212,175,55,0.18);padding:1rem;border-radius:18px;box-shadow:0 12px 24px rgba(0,0,0,0.25);margin-bottom:0.8rem;}
    [data-testid="stChatMessage"] pre {background:#121212;color:#f7f3d0;border-radius:12px;}
    .stChatInputContainer {border-top:1px solid #1c1c1e;background:rgba(8,8,9,0.9);}
    </style>
    """,
    unsafe_allow_html=True,
)


def load_secrets() -> Dict[str, Optional[str]]:
    try:
        settings = Settings(**st.secrets)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Secrets missing: {exc}")
        st.stop()

    data = settings.model_dump()
    data["QDRANT_ENABLED"] = bool(data.get("QDRANT_URL") and data.get("QDRANT_COLLECTION"))
    return data


@st.cache_resource(show_spinner=False)

def get_clients(secrets: Dict[str, Optional[str]]) -> Tuple[httpx.Client, Optional[QdrantClient]]:
    http = httpx.Client(
        base_url="https://api.openai.com/v1",
        timeout=20.0,
        headers={
            "Authorization": f"Bearer {secrets['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
        http2=True,
    )
    qc: Optional[QdrantClient] = None
    if secrets.get("QDRANT_ENABLED"):
        try:
            qc = QdrantClient(
                url=secrets["QDRANT_URL"],
                api_key=secrets.get("QDRANT_API_KEY"),
                timeout=60.0,
                prefer_grpc=True,
            )
        except Exception:  # noqa: BLE001
            st.toast("Qdrant connection failed; RAG disabled.")
    return http, qc


def retry_call(func, *args, retries: int = 4, backoff: float = 0.6, **kwargs):
    for attempt in range(1, retries + 1):
        try:
            resp = func(*args, **kwargs)
            resp.raise_for_status()
            return resp
        except (httpx.RequestError, httpx.HTTPStatusError):
            if attempt == retries:
                raise
            time.sleep(backoff * (2 ** (attempt - 1)))


def new_chat(system_prompt: str) -> Dict[str, object]:
    return {"id": str(uuid.uuid4()), "title": "Untitled chat", "messages": [], "system_prompt": system_prompt}


def ensure_state(default_prompt: str, default_model: str) -> None:

    st.session_state.setdefault("chats", [])
    st.session_state.setdefault("active_chat_id", "")
    st.session_state.setdefault("temperature", 0.3)
    st.session_state.setdefault("rag_enabled", False)
    st.session_state.setdefault("top_k", 5)
    st.session_state.setdefault("threshold", 0.2)
    st.session_state.setdefault("filters", {"ticker": "", "form": ""})
    st.session_state.setdefault("model", default_model)

    if not st.session_state["chats"]:
        chat = new_chat(default_prompt)
        st.session_state["chats"] = [chat]
        st.session_state["active_chat_id"] = chat["id"]
    elif not st.session_state["active_chat_id"]:
        st.session_state["active_chat_id"] = st.session_state["chats"][0]["id"]

def active_chat() -> Optional[Dict[str, object]]:
    cid = st.session_state.get("active_chat_id")
    for chat in st.session_state.get("chats", []):
        if chat["id"] == cid:
            return chat
    return None



@st.cache_data(show_spinner=False, ttl=60, hash_funcs={httpx.Client: lambda _: None})
def embed_query(client: httpx.Client, model: str, text: str) -> List[float]:
    payload = {"model": model, "input": text}
    resp = retry_call(client.post, "/embeddings", json=payload)
    return resp.json()["data"][0]["embedding"]


@st.cache_data(ttl=300)
def discover_facets(qc: QdrantClient, collection: str, page_size: int = 500) -> Tuple[List[str], List[str]]:
    tickers, forms = set(), set()
    next_offset = None
    while True:
        res = qc.scroll(
            collection_name=collection,
            with_payload=True,
            with_vectors=False,
            limit=page_size,
            offset=next_offset,
        )
        if isinstance(res, tuple):
            points, next_offset = res
        else:
            points = getattr(res, "points", [])
            next_offset = (
                getattr(res, "next_page_offset", None)
                or getattr(res, "offset", None)
                or getattr(res, "next_offset", None)
            )

        if not points:
            break
        for point in points:
            payload = point.payload or {}
            if (ticker := payload.get("ticker")):
                tickers.add(str(ticker))
            if (form := payload.get("form")):
                forms.add(str(form))
        if next_offset is None:
            break
    return sorted(tickers), sorted(forms)


def retrieve(
    qc: Optional[QdrantClient],
    collection: Optional[str],
    qvec: List[float],
    top_k: int,
    threshold: float,
    ticker: str,
    form: str,
) -> List[Dict[str, object]]:
    if not qc or not collection or not ticker or not form:
        return []
    flt = Filter(
        must=[
            FieldCondition(key="ticker", match=MatchValue(value=ticker)),
            FieldCondition(key="form", match=MatchValue(value=form)),
        ]
    )
    hits = qc.search(
        collection_name=collection,
        query_vector=qvec,
        limit=top_k,
        with_payload=True,
        query_filter=flt,
    )
    docs: List[Dict[str, object]] = []
    for i, h in enumerate(hits, 1):
        if threshold and h.score < threshold:
            continue
        payload = h.payload or {}
        docs.append(
            {
                "id": i,
                "text": payload.get("text", ""),
                "score": float(h.score),
                "meta": {
                    "source_path": payload.get("source_path"),
                    "chunk_idx": payload.get("chunk_idx"),
                    "ticker": payload.get("ticker"),
                    "form": payload.get("form"),
                },
            }
        )
    return docs


def build_context(docs: List[Dict[str, object]]) -> Tuple[str, List[Dict[str, object]]]:
    if not docs:
        return "", []
    limited = [doc for doc in itertools.islice((d for d in docs if d.get("text")), 5)]
    context = "\n\n".join(f"[{doc['id']}] {doc['text']}" for doc in limited)
    citations = []
    for doc in limited:
        meta = doc.get("meta", {})
        label = meta.get("source_path") or meta.get("ticker") or "Source"

        citations.append(
            {
                "id": doc["id"],
                "label": label,
                "score": doc["score"],
                "text": doc["text"],
                "meta": meta,
            }
        )
    return context, citations


def call_llm(
    client: httpx.Client,
    model: str,
    system_prompt: str,
    temperature: float,
    user_text: str,
    context: Optional[str] = None,
) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    if context:
        messages.append({"role": "system", "content": f"Context:\n{context}"})
    messages.append({"role": "user", "content": user_text})
    payload = {"model": model, "messages": messages, "temperature": temperature}
    resp = retry_call(client.post, "/chat/completions", json=payload)
    return resp.json()["choices"][0]["message"]["content"].strip()


def render_sources(sources: List[Dict[str, object]]) -> None:
    if not sources:
        return
    st.markdown("**Sources**")
    for src in sources:
        label = src.get("label") or src.get("meta", {}).get("source_path") or "Source"
        score = float(src.get("score", 0.0))
        snippet = (src.get("text") or "")[:160]
        if snippet and len(src.get("text") or "") > 160:
            snippet += "…"
        st.markdown(f"[{src['id']}] {label} (score={score:.3f}) — {snippet}")
        with st.expander(f"View source [{src['id']}]"):
            st.write(src.get("text", ""))


secrets = load_secrets()
http, qc = get_clients(secrets)

default_system_prompt = (
    "You are a helpful analyst. Use the provided context; if insufficient, say so. "
    "Cite sources with bracketed ids like [1]."
)
ensure_state(default_system_prompt, secrets.get("OPENAI_MODEL", "gpt-4o-mini"))
collection_name = secrets.get("QDRANT_COLLECTION")
if qc and collection_name:
    try:
        tickers, forms = discover_facets(qc, collection_name)
    except Exception:  # noqa: BLE001
        st.toast("Could not load Qdrant facets; you can still chat without RAG.", icon="⚠️")
        tickers, forms = [], []
else:
    tickers, forms = [], []


chat = active_chat()

col_title, col_chip = st.columns([0.8, 0.2])
with col_title:
    st.title("✨ Gold & Black Analyst")

with col_chip:
    current_model = st.session_state.get("model", secrets["OPENAI_MODEL"])
    st.markdown(
        f'<div style="text-align:right"><span class="chat-chip">{current_model}</span></div>',

        unsafe_allow_html=True,
    )

rag_available = bool(qdrant_client and secrets["qdrant_collection"])
st.session_state.setdefault("rag_toggle", rag_available)
st.session_state.setdefault("rag_top_k", 5)
st.session_state.setdefault("rag_threshold", 0.2)
st.session_state.setdefault("rag_ticker", "")
st.session_state.setdefault("rag_form", "")
st.session_state.setdefault("rag_ticker_option", "(Any)")
st.session_state.setdefault("rag_form_option", "(Any)")
st.session_state.setdefault("rag_ticker_custom", "")
st.session_state.setdefault("rag_form_custom", "")


with st.sidebar:

    if st.button("➕ New Chat", key="new_chat_btn", use_container_width=True, type="primary", help="Start fresh"):
        chat = new_chat(default_system_prompt)
        st.session_state["chats"].insert(0, chat)
        st.session_state["active_chat_id"] = chat["id"]
    st.markdown("---")
    ids = [c["id"] for c in st.session_state["chats"]]
    titles = {c["id"]: (c["title"] or "Untitled chat") for c in st.session_state["chats"]}
    if ids:
        idx = ids.index(st.session_state["active_chat_id"]) if st.session_state["active_chat_id"] in ids else 0
        selected = st.radio("Chats", ids, index=idx, format_func=lambda cid: titles.get(cid, "Untitled chat"), label_visibility="collapsed")
        if selected != st.session_state["active_chat_id"]:
            st.session_state["active_chat_id"] = selected
            chat = active_chat()
    st.markdown("---")
    model_options = list(
        dict.fromkeys(
            [
                secrets.get("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4.1-mini",
            ]
        )
    )
    current_model = st.selectbox(
        "Model",
        model_options,
        index=model_options.index(st.session_state.get("model", model_options[0])) if st.session_state.get("model") in model_options else 0,
    )
    st.session_state["model"] = current_model

    rag_toggle = st.toggle("Enable RAG (Qdrant)", value=st.session_state.get("rag_enabled", False) and bool(qc))
    st.session_state["rag_enabled"] = rag_toggle and bool(qc)
    st.session_state["top_k"] = st.slider("top_k", 3, 10, int(st.session_state["top_k"]))
    st.session_state["threshold"] = st.slider("score threshold", 0.0, 1.0, float(st.session_state["threshold"]), 0.05)
    filters = st.session_state["filters"]
    if tickers:
        ticker_options = [""] + tickers
        ticker_index = ticker_options.index(filters.get("ticker", "")) if filters.get("ticker", "") in ticker_options else 0
        filters["ticker"] = st.selectbox(
            "Ticker",
            ticker_options,
            index=ticker_index,
            format_func=lambda x: x or "Select ticker",
        )
    else:
        filters["ticker"] = st.text_input("Ticker", value=filters.get("ticker", ""))
    if forms:
        form_options = [""] + forms
        form_index = form_options.index(filters.get("form", "")) if filters.get("form", "") in form_options else 0
        filters["form"] = st.selectbox(
            "Form",
            form_options,
            index=form_index,
            format_func=lambda x: x or "Select form",
        )
    else:
        filters["form"] = st.text_input("Form", value=filters.get("form", ""))

    st.session_state["filters"] = filters
    with st.expander("Advanced", expanded=False):
        if chat:
            chat["system_prompt"] = st.text_area("System prompt", value=chat.get("system_prompt", default_system_prompt), height=120)
        st.session_state["temperature"] = st.slider("Temperature", 0.0, 1.0, float(st.session_state["temperature"]), 0.05)
    if chat:
        st.download_button(
            "Export chat (.json)",
            data=json.dumps(chat, ensure_ascii=False, indent=2),
            file_name=f"chat-{chat['id']}.json",

            mime="application/json",
            use_container_width=True,
        )

if not chat:
    st.info("Create a chat to begin.")
    st.stop()

filters = st.session_state.get("filters", {})
sel_ticker = (filters.get("ticker") or "").strip()
sel_form = (filters.get("form") or "").strip()
rag_enabled = st.session_state.get("rag_enabled", False)
has_facets = bool(sel_ticker and sel_form)
effective_rag = rag_enabled and has_facets
if rag_enabled and not has_facets:
    st.info("Select a ticker and a form to use retrieval.", icon="ℹ️")

system_prompt = chat.get("system_prompt", default_system_prompt)
temperature = float(st.session_state.get("temperature", 0.3))
top_k = int(st.session_state.get("top_k", 5))
threshold = float(st.session_state.get("threshold", 0.2))
embed_model = secrets.get("OPENAI_EMBED_MODEL", "text-embedding-3-large")
model_name = st.session_state.get("model", secrets.get("OPENAI_MODEL", "gpt-4o-mini"))

user_text = st.chat_input("Send a message")
if user_text and user_text.strip():
    message = user_text.strip()
    chat["messages"].append({"role": "user", "content": message})
    if chat["title"] == "Untitled chat":
        chat["title"] = message.splitlines()[0][:40] or "Untitled chat"

    docs: List[Dict[str, object]] = []
    citations: List[Dict[str, object]] = []
    context: Optional[str] = None
    if effective_rag:
        with st.spinner("Retrieving…"):
            try:
                qvec = embed_query(http, embed_model, message)
                docs = retrieve(qc, collection_name, qvec, top_k, threshold, sel_ticker, sel_form)
                context, citations = build_context(docs)
            except Exception:  # noqa: BLE001
                st.toast("RAG unavailable; continuing without context.", icon="⚠️")
                docs, citations, context = [], [], None

    with st.spinner("Generating…"):
        try:
            reply = call_llm(
                http,
                model_name,
                system_prompt,
                temperature,
                message,
                context=context if effective_rag and context else None,
            )
        except Exception:  # noqa: BLE001
            st.toast("⚠️ Generation failed. Please retry.")
        else:
            assistant_entry: Dict[str, object] = {"role": "assistant", "content": reply}
            if citations:
                assistant_entry["meta"] = {"sources": citations}
            chat["messages"].append(assistant_entry)

for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(msg.get("meta", {}).get("sources", []))

