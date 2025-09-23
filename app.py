import json
import time
import uuid
from typing import Dict, List, Optional, Tuple

import httpx
import streamlit as st
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


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
def get_clients(config: Dict[str, Optional[str]]) -> Tuple[httpx.Client, Optional[QdrantClient]]:
    client = httpx.Client(
        base_url="https://api.openai.com/v1",
        headers={"Authorization": f"Bearer {config['OPENAI_API_KEY']}", "Content-Type": "application/json"},
        timeout=httpx.Timeout(15.0, connect=5.0, read=15.0),
        http2=True,
    )
    qclient: Optional[QdrantClient] = None
    if config.get("QDRANT_ENABLED"):
        try:
            qclient = QdrantClient(
                url=config["QDRANT_URL"],
                api_key=config.get("QDRANT_API_KEY"),
                prefer_grpc=True,
                timeout=5.0,
            )
        except Exception:  # noqa: BLE001
            st.toast("Qdrant connection failed; RAG disabled.")
    return client, qclient


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


def ensure_state(default_prompt: str) -> None:
    st.session_state.setdefault("chats", [])
    st.session_state.setdefault("active_chat_id", "")
    st.session_state.setdefault("temperature", 0.3)
    st.session_state.setdefault("rag_enabled", False)
    st.session_state.setdefault("top_k", 5)
    st.session_state.setdefault("threshold", 0.2)
    st.session_state.setdefault("filters", {"ticker": "", "form": ""})
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


def build_filter(ticker: str, form: str) -> Optional[qmodels.Filter]:
    conds = []
    if ticker:
        conds.append(qmodels.FieldCondition(key="ticker", match=qmodels.MatchValue(value=ticker)))
    if form:
        conds.append(qmodels.FieldCondition(key="form", match=qmodels.MatchValue(value=form)))
    return qmodels.Filter(must=conds) if conds else None


def retrieve(
    qc: Optional[QdrantClient],
    collection: Optional[str],
    vector: List[float],
    top_k: int,
    threshold: float,
    ticker: str,
    form: str,
) -> List[Dict[str, object]]:
    if not qc or not collection:
        return []
    results = qc.search(
        collection_name=collection,
        query_vector=vector,
        limit=top_k,
        with_payload=True,
        score_threshold=None,
        query_filter=build_filter(ticker, form),
    )
    docs: List[Dict[str, object]] = []
    for idx, point in enumerate(results, 1):
        score = float(point.score or 0.0)
        if score < threshold:
            continue
        payload = point.payload or {}
        docs.append(
            {
                "id": idx,
                "text": payload.get("text") or payload.get("chunk") or "",
                "score": score,
                "meta": {
                    "source": payload.get("source") or payload.get("path") or payload.get("ticker") or "Document",
                    "chunk": payload.get("chunk_id") or payload.get("chunk"),
                },
            }
        )
    return docs[:5]


def build_context(docs: List[Dict[str, object]]) -> Tuple[str, List[Dict[str, object]]]:
    if not docs:
        return "", []
    context = "\n\n".join(f"[{doc['id']}] {doc['text']}" for doc in docs if doc["text"])
    cites = [
        {
            "id": doc["id"],
            "label": doc["meta"]["source"],
            "score": doc["score"],
            "text": doc["text"],
        }
        for doc in docs
    ]
    return context, cites


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
        snippet = src["text"][:140] + ("…" if len(src["text"]) > 140 else "")
        st.markdown(f"[{src['id']}] {src['label']} ({src['score']:.2f}) — {snippet}")
        with st.expander(f"View source [{src['id']}]"):
            st.write(src["text"])


secrets = load_secrets()
client, qclient = get_clients(secrets)
default_system_prompt = (
    "You are a helpful analyst. Use the provided context; if insufficient, say so. "
    "Cite sources with bracketed ids like [1]."
)
ensure_state(default_system_prompt)
chat = active_chat()

col_title, col_chip = st.columns([0.8, 0.2])
with col_title:
    st.title("✨ Gold & Black Analyst")
with col_chip:
    st.markdown(
        f'<div style="text-align:right"><span class="chat-chip">{secrets["OPENAI_MODEL"]}</span></div>',
        unsafe_allow_html=True,
    )

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
    rag_toggle = st.toggle("Enable RAG (Qdrant)", value=st.session_state.get("rag_enabled", False) and bool(qclient))
    st.session_state["rag_enabled"] = rag_toggle and bool(qclient)
    st.session_state["top_k"] = st.slider("top_k", 3, 10, int(st.session_state["top_k"]))
    st.session_state["threshold"] = st.slider("score threshold", 0.0, 1.0, float(st.session_state["threshold"]), 0.05)
    filters = st.session_state["filters"]
    filters["ticker"] = st.text_input("Ticker", value=filters.get("ticker", ""))
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

for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(msg.get("meta", {}).get("sources", []))

if prompt := st.chat_input("Ask anything"):
    chat["messages"].append({"role": "user", "content": prompt})
    if chat["title"] == "Untitled chat":
        chat["title"] = prompt.splitlines()[0][:40] or "Untitled chat"

    citations: List[Dict[str, object]] = []
    context: Optional[str] = None
    if st.session_state["rag_enabled"]:
        with st.spinner("Retrieving…"):
            try:
                vector = embed_query(client, secrets["OPENAI_EMBED_MODEL"], prompt)
                docs = retrieve(
                    qclient,
                    secrets.get("QDRANT_COLLECTION"),
                    vector,
                    st.session_state["top_k"],
                    st.session_state["threshold"],
                    st.session_state["filters"].get("ticker", ""),
                    st.session_state["filters"].get("form", ""),
                )
                context, citations = build_context(docs)
            except Exception:  # noqa: BLE001
                st.toast("RAG unavailable; continuing without context.")
                context, citations = None, []
    with st.spinner("Generating…"):
        try:
            reply = call_llm(
                client,
                secrets["OPENAI_MODEL"],
                chat.get("system_prompt", default_system_prompt),
                float(st.session_state["temperature"]),
                prompt,
                context=context,
            )
        except Exception:  # noqa: BLE001
            st.toast("⚠️ Generation failed. Please retry.")
        else:
            chat["messages"].append({"role": "assistant", "content": reply, "meta": {"sources": citations}})
            with st.chat_message("assistant"):
                st.markdown(reply)
                render_sources(citations)
