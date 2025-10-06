# 10k Chatbot Frontend

Modern Streamlit interface for exploring Gemini or Groq chat models with a luxe gold-and-black theme. The app keeps multiple conversations in memory, lets you fine-tune prompts, and exports transcripts for later review.

## Features

- ⚡️ **Multi-chat session state** with quick switching between previous conversations.
- 🧠 **Custom system prompt & temperature controls** in a compact “Advanced” panel.
- 🔁 **Reliable Gemini / Groq requests** powered by a cached `httpx` client and lightweight retry logic.
- 📚 **Gemini `gemini-embedding-001` embeddings** drive Qdrant retrieval for accurate, cited answers.
- 💬 **Polished chat bubbles** built with Streamlit’s native chat components and custom CSS styling.
- 📦 **One-click JSON export** of the active conversation for downstream analysis.

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`
- Google AI Studio (Gemini) API access for chat + embeddings (`gemini-embedding-001`, 768-dim vectors)
- (Optional) Groq API access for additional chat models

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Add your credentials to Streamlit secrets (recommended) or environment variables. For local development, create `.streamlit/secrets.toml` in the project root:

```toml
[default]
GEMINI_API_KEY = "AIza..."
GEMINI_MODEL = "gemini-1.5-flash"          # optional override
# Embeddings and Qdrant
GEMINI_EMBED_MODEL = "gemini-embedding-001" # optional override, returns 768-dim vectors
QDRANT_URL = "https://YOUR-QDRANT-ENDPOINT"
QDRANT_API_KEY = "qdrant-..."
QDRANT_COLLECTION = "sec_filings"
# Set if your collection uses named vectors
QDRANT_VECTOR_NAME = "text_vector"
# Optional Groq provider support
GROQ_API_KEY = "gsk-..."
GROQ_MODEL = "llama-3.1-8b-instant"
```

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
