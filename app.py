import json
import time
import uuid
from typing import Dict, List, Optional, Tuple

import httpx
import streamlit as st
from pydantic import BaseModel, Field

DEFAULT_SYSTEM_PROMPT = "You are a concise, thoughtful assistant who replies with clarity and warmth."
DEFAULT_TEMPERATURE = 0.3
MAX_TITLE_LEN = 40


st.set_page_config(page_title="Gold & Black Chat", page_icon="✨", layout="wide")

APP_CSS = """
<style>
.stApp {background-color:#0b0b0c; color:#f4e9c9;}
section[data-testid="stSidebar"] {background:linear-gradient(180deg,#111112 0%,#050506 100%); border-right:1px solid #2a2a2d;}
section[data-testid="stSidebar"] .stButton button, section[data-testid="stSidebar"] .stDownloadButton button {width:100%; border-radius:18px; padding:0.6rem 1rem; background:rgba(212,175,55,0.12); border:1px solid rgba(212,175,55,0.35); color:#f5ddb2; font-weight:600; box-shadow:0 8px 16px rgba(0,0,0,0.25);}
section[data-testid="stSidebar"] .stButton button:hover, section[data-testid="stSidebar"] .stDownloadButton button:hover {background:rgba(212,175,55,0.25); border-color:#d4af37;}
section[data-testid="stSidebar"] div[role="radiogroup"] {display:flex; flex-direction:column; gap:0.4rem;}
section[data-testid="stSidebar"] div[role="radiogroup"] label {width:100%; background:rgba(255,255,255,0.02); border:1px solid rgba(212,175,55,0.12); padding:0.55rem 0.9rem; border-radius:14px; color:#d9c68b; box-shadow:0 6px 18px rgba(0,0,0,0.25); transition:all 0.2s ease;}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {border-color:#d4af37; color:#ffe9ad;}
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {background:rgba(212,175,55,0.18); border-color:#d4af37; color:#ffe9ad;}
.chat-chip {display:inline-flex; align-items:center; gap:0.35rem; background:rgba(212,175,55,0.18); border-radius:999px; padding:0.25rem 0.9rem; font-size:0.75rem; color:#f7df9d; border:1px solid rgba(212,175,55,0.35); box-shadow:0 4px 10px rgba(0,0,0,0.25);}
[data-testid="stHeader"] {background:transparent;}
.stMarkdown h1 {color:#f5ddb2;}
.block-container {padding-top:1.2rem;}
[data-testid="stChatMessage"] {background:rgba(255,255,255,0.03); border:1px solid rgba(212,175,55,0.2); padding:1rem; border-radius:18px; box-shadow:0 12px 24px rgba(0,0,0,0.25); margin-bottom:0.8rem;}
[data-testid="stChatMessage"] pre {background:#121212; color:#f7f3d0; border-radius:12px;}
[data-testid="stChatMessage-avatar"] {background:rgba(212,175,55,0.3);}
.stChatInputContainer {border-top:1px solid #1c1c1e; background:rgba(8,8,9,0.85);}
.stSlider > div[data-baseweb="slider"] {color:#f5ddb2;}
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)


class ChatMessage(BaseModel):
    role: str
    content: str


class Chat(BaseModel):
    id: str
    title: str = "Untitled chat"
    messages: List[ChatMessage] = Field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.messages.append(ChatMessage(role=role, content=content))


def load_secrets() -> Dict[str, str]:
    try:
        api_key = st.secrets["OPENAI_API_KEY"].strip()
    except KeyError:
        st.error("Please add OPENAI_API_KEY to your Streamlit secrets to chat.")
        st.stop()
    model = str(st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")).strip()
    return {"api_key": api_key, "model": model}


@st.cache_resource(show_spinner=False)
def get_client(api_key: str) -> httpx.Client:
    return httpx.Client(
        base_url="https://api.openai.com/v1",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=httpx.Timeout(15.0, connect=5.0, read=15.0),
        http2=True,
    )


def retry_call(func, *args, retries: int = 4, backoff: float = 0.6, **kwargs):
    for attempt in range(1, retries + 1):
        try:
            response = func(*args, **kwargs)
            response.raise_for_status()
            return response
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            if attempt == retries:
                raise exc
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


secrets = load_secrets()
client = get_client(secrets["api_key"])
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

with st.sidebar:
    if st.button("➕ New Chat", use_container_width=True):
        chat = create_chat(DEFAULT_SYSTEM_PROMPT)
        st.session_state["chats"].insert(0, chat.model_dump())
        set_active_chat(chat.id)
        active_idx, active_chat = 0, chat
    st.markdown("---")
    chat_ids = [chat["id"] for chat in st.session_state["chats"]]
    if chat_ids:
        titles = {chat["id"]: trim_title(chat["title"]) for chat in st.session_state["chats"]}
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

    if prompt := st.chat_input("Send a message"):
        active_chat.add("user", prompt)
        if active_chat.title == "Untitled chat":
            active_chat.title = trim_title(prompt)
        if active_idx is not None:
            persist_chat(active_idx, active_chat)

        messages_payload: List[Dict[str, str]] = []
        system_prompt = st.session_state["system_prompts"].get(active_chat.id, DEFAULT_SYSTEM_PROMPT)
        if system_prompt:
            messages_payload.append({"role": "system", "content": system_prompt})
        messages_payload.extend(msg.model_dump() for msg in active_chat.messages)

        payload = {
            "model": secrets["model"] or "gpt-4o-mini",
            "temperature": float(st.session_state["temperature"]),
            "messages": messages_payload,
        }

        try:
            response = retry_call(client.post, "/chat/completions", json=payload)
            data = response.json()
            reply = data["choices"][0]["message"]["content"].strip()
            active_chat.add("assistant", reply)
            if active_idx is not None:
                persist_chat(active_idx, active_chat)
        except Exception:  # noqa: BLE001
            st.toast("⚠️ Sorry, something went wrong. Please try again.")
else:
    st.info("Create a chat to begin.")
