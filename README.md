# DocuMind — RAG Chatbot

A RAG (Retrieval-Augmented Generation) chatbot built with **FastAPI** and a single-page UI. Upload documents and ask questions — all answers are grounded in your uploaded files.

## Project Structure

```
documind/
├── app.py          # FastAPI backend — document processing, FAISS, RAG, all API routes
├── index.html      # Single-page frontend — served by FastAPI, CSS + JS fully inline
├── requirements.txt
├── .env            # Your API key goes here (copy from .env.example)
├── .env.example
└── vector_store/   # Auto-created at runtime — stores FAISS index + document metadata
```

## Tech Stack

| Layer | Tool |
|---|---|
| Backend | FastAPI + Uvicorn |
| Document Parsing | pdfplumber (PDF), python-docx (DOCX), built-in (TXT) |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector Store | FAISS (persisted to disk) |
| LLM | Groq — `llama-3.3-70b-versatile` |
| Frontend | Vanilla HTML + CSS + JS (no frameworks) |

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd documind
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

```bash
cp .env.example .env
```

Open `.env` and set your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

### 5. Run the server

```bash
python app.py
```

Open **http://localhost:8000** in your browser.

## How It Works

### Upload
- Drag and drop or browse to upload a PDF, DOCX, or TXT file
- The backend extracts text, splits it into chunks (800 chars, 150 overlap), embeds each chunk using `all-MiniLM-L6-v2`, and stores the vectors in FAISS
- The FAISS index is saved to disk — uploaded documents survive server restarts

### Ask Questions
- Type a question and press **Send** or **Enter**
- The backend retrieves the top 5 most relevant chunks from the vector store
- Those chunks plus the last 10 conversation turns are sent to Groq (LLaMA 3)
- The answer is returned with the source filenames cited

### Chat History
- Conversation history is kept per session (identified by a session ID stored in `localStorage`)
- The last 10 messages are sent to the LLM on every request so follow-up questions work correctly
- Use **Clear Chat** in the sidebar to reset the current session

### Document Management
- The sidebar lists all indexed documents
- Click **✕** next to any document to remove it — the FAISS index is rebuilt automatically

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the UI |
| `POST` | `/upload` | Upload and index a document |
| `POST` | `/chat` | Ask a question (RAG) |
| `GET` | `/documents` | List all indexed documents |
| `DELETE` | `/documents/{filename}` | Remove a document |
| `DELETE` | `/chat/{session_id}` | Clear chat history for a session |

## Supported File Types

- `.pdf` — extracted with pdfplumber (page by page)
- `.docx` — extracted with python-docx (paragraph by paragraph)
- `.txt` — read as UTF-8 text

## Configuration

These values can be overridden via environment variables in `.env`:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Required. Your Groq API key |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `MAX_TOKENS` | `1024` | Max tokens in the LLM response |
