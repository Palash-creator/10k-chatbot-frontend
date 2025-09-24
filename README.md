# 10k Chatbot Frontend

Modern Streamlit interface for exploring OpenAI chat models with a luxe gold-and-black theme. The app keeps multiple conversations in memory, lets you fine-tune prompts, and exports transcripts for later review.

## Features

- ⚡️ **Multi-chat session state** with quick switching between previous conversations.
- 🧠 **Custom system prompt & temperature controls** in a compact “Advanced” panel.
- 🔁 **Reliable OpenAI requests** powered by a cached `httpx` client and lightweight retry logic.
- 💬 **Polished chat bubbles** built with Streamlit’s native chat components and custom CSS styling.
- 📦 **One-click JSON export** of the active conversation for downstream analysis.

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`
- OpenAI API access (Chat Completions endpoint)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Add your credentials to Streamlit secrets (recommended) or environment variables. For local development, create `.streamlit/secrets.toml` in the project root:

```toml
[default]
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4o-mini"  # optional override
```

`OPENAI_MODEL` defaults to `gpt-4o-mini` when omitted.

## Run the App

```bash
streamlit run app.py
```

Streamlit will open a browser window at `http://localhost:8501`.

## Usage Tips

1. Click **➕ New Chat** to start a fresh conversation; prior chats stay available in the sidebar.
2. Use the **Advanced** expander to adjust the system prompt or temperature before sending a message.
3. Download the active conversation anytime with **Export chat (.json)**.
4. Secrets remain on the server—no keys are stored in the repository.

Deploy to Streamlit Community Cloud or your own infrastructure once secrets are configured.
