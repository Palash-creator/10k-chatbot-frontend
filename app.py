import json
import time
import uuid
from typing import Dict, List, Optional, Tuple

import httpx
import streamlit as st
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful analyst. Use the provided context when available; if it is insufficient, say so. "
    "Always cite sources with bracketed ids like [1]."
)
DEFAULT_TEMPERATURE = 0.3
MAX_TITLE_LEN = 48


st.set_page_config(page_title="Gold & Black Chat", page_icon="✨", layout="wide")

APP_CSS = """
<style>
.stApp {background:#0b0b0c; color:#f4e9c9;}
section[data-testid="stSidebar"] {background:linear-gradient(180deg,#111112 0%,#050506 100%); border-right:1px solid #262628;}
section[data-testid="stSidebar"] .stButton button, section[data-testid="stSidebar"] .stDownloadButton button {
    width:100%; border-radius:18px; padding:0.55rem 0.9rem; background:rgba(212,175,55,0.15);
    border:1px solid rgba(212,175,55,0.4); color:#f5ddb2; font-weight:600; box-shadow:0 10px 24px rgba(0,0,0,0.35);
}
section[data-testid="stSidebar"] .stButton button:hover, section[data-testid="stSidebar"] .stDownloadButton button:hover {
    background:rgba(212,175,55,0.32); border-color:#d4af37;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background:rgba(255,255,255,0.02); border:1px solid rgba(212,175,55,0.14); padding:0.55rem 0.9rem;
    border-radius:14px; color:#d9c68b; transition:all 0.2s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {background:rgba(212,175,55,0.2); color:#ffe9ad; border-color:#d4af37;}
.chat-chip {display:inline-flex; align-items:center; gap:0.35rem; background:rgba(212,175,55,0.18); border-radius:999px; padding:0.25rem 0.9rem;
    font-size:0.75rem; color:#f7df9d; border:1px solid rgba(212,175,55,0.32); box-shadow:0 6px 14px rgba(0,0,0,0.35);
}
[data-testid="stHeader"] {background:transparent;}
.block-container {padding-top:1.2rem;}
[data-testid="stChatMessage"] {background:rgba(255,255,255,0.04); border:1px solid rgba(212,175,55,0.24); padding:1rem; border-radius:18px; box-shadow:0 14px 32px rgba(0,0,0,0.45); margin-bottom:0.85rem;}
[data-testid="stChatMessage"] pre {background:#111214; color:#f5f1d8; border-radius:12px;}
[data-testid="stChatMessage-avatar"] {background:rgba(212,175,55,0.28);}
.stChatInputContainer {border-top:1px solid #1c1c1e; background:rgba(8,8,9,0.9);} 
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)


class ChatMessage(BaseModel):
    role: str
    content: str
    meta: Optional[Dict] = None


class Chat(BaseModel):
    id: str
    title: str = "Untitled chat"
    messages: List[ChatMessage] = Field(default_factory=list)

    def add(self, role: str, content: str, meta: Optional[Dict] = None) -> None:
        self.messages.append(ChatMessage(role=role, content=content, meta=meta))


def load_secrets() -> Dict[str, str]:
    try:
        api_key = st.secrets["OPENAI_API_KEY"].strip()
    except KeyError:
        st.error("Please add OPENAI_API_KEY to your Streamlit secrets to chat.")
        st.stop()
    model = str(st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")).strip()
    embed_model = str(st.secrets.get("OPENAI_EMBED_MODEL", "text-embedding-3-large")).strip()
    return {
        "api_key": api_key,
        "model": model,
        "embed_model": embed_model,
        "qdrant_url": st.secrets.get("QDRANT_URL", "").strip(),
        "qdrant_key": st.secrets.get("QDRANT_API_KEY", "").strip(),
        "qdrant_collection": st.secrets.get("QDRANT_COLLECTION", "").strip(),
    }


@st.cache_resource(show_spinner=False)
def get_clients(secrets: Dict[str, str]) -> Tuple[httpx.Client, Optional[QdrantClient]]:
    http_client = httpx.Client(
        base_url="https://api.openai.com/v1",
        headers={
            "Authorization": f"Bearer {secrets['api_key']}",
            "Content-Type": "application/json",
        },
        timeout=httpx.Timeout(20.0, connect=5.0, read=20.0),
        http2=True,
    )
    qc: Optional[QdrantClient] = None
    if secrets["qdrant_url"] and secrets["qdrant_collection"]:
        try:
            qc = QdrantClient(url=secrets["qdrant_url"], api_key=secrets["qdrant_key"], prefer_grpc=True)
        except Exception:
            qc = None
    return http_client, qc



def retry_call(func, *args, retries: int = 4, backoff: float = 0.6, **kwargs):
    for attempt in range(1, retries + 1):
        try:
            response = func(*args, **kwargs)
            response.raise_for_status()
            return response

        except (httpx.RequestError, httpx.HTTPStatusError):
            if attempt == retries:
                raise
            time.sleep(backoff * (2 ** (attempt - 1)))

def ensure_state(default_prompt: str) -> None:
    st.session_state.setdefault("chats", [])
    st.session_state.setdefault("active_chat_id", None)
    st.session_state.setdefault("system_prompts", {})
    st.session_state.setdefault("temperature", DEFAULT_TEMPERATURE)
    if not st.session_state["chats"]:
        chat = create_chat(default_prompt)
        st.session_state["chats"].append(chat.model_dump())
        st.session_state["active_chat_id"] = chat.id
    elif st.session_state["active_chat_id"] is None:
        st.session_state["active_chat_id"] = st.session_state["chats"][0]["id"]
    for chat in st.session_state["chats"]:
        st.session_state["system_prompts"].setdefault(chat["id"], default_prompt)


def create_chat(default_prompt: str) -> Chat:
    chat = Chat(id=str(uuid.uuid4()))
    st.session_state.setdefault("system_prompts", {})[chat.id] = default_prompt
    return chat


def set_active_chat(chat_id: str) -> None:
    st.session_state["active_chat_id"] = chat_id


def get_active_chat_index() -> Optional[int]:
    cid = st.session_state.get("active_chat_id")
    for idx, chat in enumerate(st.session_state.get("chats", [])):
        if chat["id"] == cid:
            return idx
    return None


def load_active_chat() -> Tuple[Optional[int], Optional[Chat]]:
    idx = get_active_chat_index()
    if idx is None:
        return None, None
    chat = Chat.model_validate(st.session_state["chats"][idx])
    return idx, chat


def persist_chat(idx: int, chat: Chat) -> None:
    st.session_state["chats"][idx] = chat.model_dump()


def trim_title(text: str, limit: int = MAX_TITLE_LEN) -> str:
    stripped = text.strip().splitlines()[0] if text.strip() else "Untitled chat"
    return (stripped[: limit - 1] + "…") if len(stripped) > limit else stripped


@st.cache_data(show_spinner=False, ttl=60)
def _cached_embedding(model: str, text: str) -> List[float]:
    response = retry_call(http_client.post, "/embeddings", json={"model": model, "input": text})
    return response.json()["data"][0]["embedding"]


def embed_query(client: httpx.Client, model: str, text: str) -> List[float]:
    return _cached_embedding(model, text)



def retrieve(
    qc: Optional[QdrantClient],
    collection: str,
    qvec: List[float],

    top_k: int,
    threshold: float,
    ticker: str,
    form: str,
) -> List[Dict]:
    if not qc or not collection:
        return []
    try:
        conditions = []
        if ticker:
            conditions.append(FieldCondition(key="ticker", match=MatchValue(value=ticker)))
        if form:
            conditions.append(FieldCondition(key="form", match=MatchValue(value=form)))
        query_filter = Filter(must=conditions) if conditions else None
        results = qc.search(
            collection_name=collection,
            query_vector=qvec,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )
        docs = []
        for idx, hit in enumerate(results, 1):
            if threshold and hit.score < threshold:
                continue
            payload = hit.payload or {}
            text = (payload.get("text") or payload.get("chunk") or "").strip()
            if not text:
                continue
            docs.append(
                {
                    "id": idx,
                    "text": text,
                    "score": float(hit.score),
                    "meta": payload,
                }
            )
        return docs[:top_k]
    except Exception:
        st.toast("⚠️ RAG retrieval failed; continuing without context.")
        return []


def build_context(docs: List[Dict]) -> Tuple[str, List[Dict]]:
    if not docs:
        return "", []
    chunks, citations = [], []
    for doc in docs:
        snippet = doc["text"][:900].strip()
        chunks.append(f"[{doc['id']}] {snippet}")
        label = doc["meta"].get("source_path") or doc["meta"].get("ticker") or doc["meta"].get("id") or f"Doc {doc['id']}"
        citations.append(
            {
                "id": doc["id"],
                "label": label,
                "score": doc["score"],
                "text": snippet,
            }
        )
    return "\n\n".join(chunks), citations


def call_llm(
    client: httpx.Client,
    model: str,
    system_prompt: str,
    temperature: float,
    user_text: str,
    context: Optional[str] = None,
) -> str:

    messages = [{"role": "system", "content": system_prompt.strip() or DEFAULT_SYSTEM_PROMPT}]
    if context:
        messages.append({"role": "system", "content": f"Context:\n{context}"})
    messages.append({"role": "user", "content": user_text})
    payload = {"model": model or "gpt-4o-mini", "temperature": temperature, "messages": messages}
    response = retry_call(client.post, "/chat/completions", json=payload)
    return response.json()["choices"][0]["message"]["content"].strip()


secrets = load_secrets()
http_client, qdrant_client = get_clients(secrets)
if secrets["qdrant_url"] and not qdrant_client:
    st.toast("⚠️ Qdrant unavailable; RAG disabled.")

ensure_state(DEFAULT_SYSTEM_PROMPT)
active_idx, active_chat = load_active_chat()

col_title, col_chip = st.columns([0.8, 0.2])
with col_title:
    st.title("✨ Gold & Black Chat")
with col_chip:
    st.markdown(
        f'<div style="text-align:right"><span class="chat-chip">{secrets["model"]}</span></div>',
        unsafe_allow_html=True,
    )

rag_available = bool(qdrant_client and secrets["qdrant_collection"])
st.session_state.setdefault("rag_toggle", rag_available)
st.session_state.setdefault("rag_top_k", 5)
st.session_state.setdefault("rag_threshold", 0.2)
st.session_state.setdefault("rag_ticker", "")
st.session_state.setdefault("rag_form", "")

with st.sidebar:
    if st.button("➕ New Chat", use_container_width=True):
        chat = create_chat(DEFAULT_SYSTEM_PROMPT)
        st.session_state["chats"].insert(0, chat.model_dump())
        set_active_chat(chat.id)
        active_idx, active_chat = 0, chat
    st.markdown("---")
    chat_ids = [chat["id"] for chat in st.session_state["chats"]]
    if chat_ids:
        titles = {chat["id"]: trim_title(chat.get("title", "Untitled chat")) for chat in st.session_state["chats"]}
        current_idx = chat_ids.index(st.session_state["active_chat_id"]) if st.session_state["active_chat_id"] in chat_ids else 0
        selected = st.radio(
            "Chats",
            chat_ids,
            index=current_idx,
            format_func=lambda cid: titles.get(cid, "Untitled chat"),
            label_visibility="collapsed",
            key="chat_selector",
        )
        if selected != st.session_state["active_chat_id"]:
            set_active_chat(selected)
            active_idx, active_chat = load_active_chat()
    st.markdown("---")
    st.session_state["rag_toggle"] = st.toggle(
        "Enable RAG (Qdrant)",
        value=st.session_state["rag_toggle"],
        disabled=not rag_available,
    )
    if st.session_state["rag_toggle"] and rag_available:
        st.session_state["rag_top_k"] = st.slider("Top K", 3, 10, st.session_state["rag_top_k"])
        st.session_state["rag_threshold"] = st.slider("Score threshold", 0.0, 1.0, float(st.session_state["rag_threshold"]), 0.05)
        st.session_state["rag_ticker"] = st.text_input("Ticker filter", st.session_state["rag_ticker"])
        st.session_state["rag_form"] = st.text_input("Form filter", st.session_state["rag_form"])
    with st.expander("Advanced", expanded=False):
        if active_chat:
            prompt_key = active_chat.id
            prompt_value = st.session_state["system_prompts"].get(prompt_key, DEFAULT_SYSTEM_PROMPT)
            updated_prompt = st.text_area("System prompt", value=prompt_value, height=120)
            st.session_state["system_prompts"][prompt_key] = updated_prompt.strip() or DEFAULT_SYSTEM_PROMPT
        st.session_state["temperature"] = st.slider("Temperature", 0.0, 1.0, float(st.session_state["temperature"]), 0.05)
    if active_chat:
        export_data = json.dumps(active_chat.model_dump(), ensure_ascii=False, indent=2)
        st.download_button(
            "Export chat (.json)",
            data=export_data,
            file_name=f"chat-{active_chat.id}.json",
            mime="application/json",
            use_container_width=True,
        )

active_idx, active_chat = load_active_chat()

if active_chat:
    for msg in active_chat.messages:
        with st.chat_message(msg.role):
            st.markdown(msg.content)
            if msg.meta and msg.meta.get("sources"):
                st.markdown("**Sources**")
                for src in msg.meta["sources"]:
                    st.markdown(f"[{src['id']}] {src['label']} (score {src['score']:.2f})")
                    with st.expander(f"View snippet [{src['id']}]"):
                        st.markdown(src["text"])

    user_prompt = st.chat_input("Send a message")
    if user_prompt:
        active_chat.add("user", user_prompt)
        if active_chat.title == "Untitled chat":
            active_chat.title = trim_title(user_prompt)
        if active_idx is not None:
            persist_chat(active_idx, active_chat)

        system_prompt = st.session_state["system_prompts"].get(active_chat.id, DEFAULT_SYSTEM_PROMPT)
        temperature = float(st.session_state["temperature"])

        context_text = ""
        citations: List[Dict] = []
        if st.session_state["rag_toggle"] and rag_available:
            with st.spinner("Retrieving context..."):
                try:
                    embedding = embed_query(http_client, secrets["embed_model"], user_prompt)
                    docs = retrieve(
                        qdrant_client,
                        secrets["qdrant_collection"],
                        embedding,
                        st.session_state["rag_top_k"],
                        st.session_state["rag_threshold"],
                        st.session_state["rag_ticker"].strip(),
                        st.session_state["rag_form"].strip(),
                    )
                except Exception:
                    docs = []
                    st.toast("⚠️ Retrieval failed; generating without context.")
            context_text, citations = build_context(docs)

        history_lines = []
        for past in active_chat.messages[:-1]:
            prefix = "User" if past.role == "user" else "Assistant"
            history_lines.append(f"{prefix}: {past.content}")
        conversation = "\n".join(history_lines)
        latest = active_chat.messages[-1].content
        user_payload = f"{conversation}\nUser: {latest}" if conversation else latest

        with st.spinner("Generating answer..."):
            try:
                reply = call_llm(
                    http_client,
                    secrets["model"],
                    system_prompt,
                    temperature,
                    user_payload,
                    context=context_text or None,
                )
            except Exception:
                st.toast("⚠️ Sorry, something went wrong. Please try again.")
                reply = None
        if reply:
            meta = {"sources": citations} if citations else None
            active_chat.add("assistant", reply, meta=meta)
            if active_idx is not None:
                persist_chat(active_idx, active_chat)
else:
    st.info("Create a chat to begin.")

