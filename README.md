# DocuMind — RAG Chatbot (Single-File Edition)

A RAG chatbot in just **2 files**: `app.py` (backend) + `index.html` (frontend).

## Files

| File | What it is |
|------|-----------|
| `app.py` | FastAPI backend — document processing, FAISS vector store, Groq RAG, all routes |
| `index.html` | Single-page UI — served by FastAPI, all CSS + JS inline |
| `requirements.txt` | Python dependencies |
| `.env.example` | Copy to `.env` and add your Groq API key |

## Setup

```bash
# 1. Create and activate virtualenv
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env → set GROQ_API_KEY=your_key_here

# 4. Run
python app.py
```

Open **http://localhost:8000** in your browser.

## How It Works

1. **Upload** PDF/DOCX/TXT → text extracted, split into chunks, embedded with `all-MiniLM-L6-v2`, stored in FAISS (persisted to `vector_store/`)
2. **Ask** a question → top-5 relevant chunks retrieved → sent to Groq (LLaMA 3) with conversation history → answer streamed back
3. **Context** — last 10 conversation turns are included in every request for follow-up question support
