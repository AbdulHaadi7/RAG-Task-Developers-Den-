# DocuMind - my RAG chatbot backend
# just run: python app.py

import io
import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import pdfplumber
from docx import Document as DocxDocument

import faiss
from sentence_transformers import SentenceTransformer

from langchain.text_splitter import RecursiveCharacterTextSplitter

from groq import Groq

import uvicorn

# --- config stuff ---
GROQ_API_KEY = "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"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K          = int(os.getenv("TOP_K", "5"))
MAX_TOKENS     = int(os.getenv("MAX_TOKENS", "1024"))
CHUNK_SIZE     = 800
CHUNK_OVERLAP  = 150
EMBED_MODEL    = "all-MiniLM-L6-v2"
VECTOR_DIR     = "vector_store"
os.makedirs(VECTOR_DIR, exist_ok=True)

# --- document processing (extract text, split into chunks) ---
@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

def extract_text(content: bytes, ext: str) -> str:
    if ext == ".pdf":
        pages = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages.append(t)
        return "\n\n".join(pages)
    elif ext == ".docx":
        doc = DocxDocument(io.BytesIO(content))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    elif ext == ".txt":
        return content.decode("utf-8", errors="replace")
    raise ValueError(f"Unsupported extension: {ext}")

def process_document(content: bytes, ext: str, filename: str) -> List[Chunk]:
    text = extract_text(content, ext).strip()
    if not text:
        raise ValueError("No text could be extracted from the document.")
    return [Chunk(text=c, source=filename, chunk_index=i)
            for i, c in enumerate(splitter.split_text(text))]

# --- vector store (faiss index + metadata saved to disk) ---
embed_model = SentenceTransformer(EMBED_MODEL)
DIM = embed_model.get_sentence_embedding_dimension()

_index: faiss.IndexFlatL2 = faiss.IndexFlatL2(DIM)
_metadata: List[Dict[str, Any]] = []

def _embed(texts: List[str]) -> np.ndarray:
    return embed_model.encode(texts, show_progress_bar=False,
                              convert_to_numpy=True).astype("float32")

def _save_store():
    faiss.write_index(_index, os.path.join(VECTOR_DIR, "index.faiss"))
    with open(os.path.join(VECTOR_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(_metadata, f)

def _load_store():
    global _index, _metadata
    ip = os.path.join(VECTOR_DIR, "index.faiss")
    mp = os.path.join(VECTOR_DIR, "metadata.pkl")
    if os.path.exists(ip) and os.path.exists(mp):
        _index = faiss.read_index(ip)
        with open(mp, "rb") as f:
            _metadata = pickle.load(f)

def add_to_store(chunks: List[Chunk]) -> int:
    _index.add(_embed([c.text for c in chunks]))
    for c in chunks:
        _metadata.append({"text": c.text, "source": c.source, "chunk_index": c.chunk_index})
    _save_store()
    return len(chunks)

def search_store(query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    if _index.ntotal == 0:
        return []
    k = min(k, _index.ntotal)
    _, indices = _index.search(_embed([query]), k)
    return [_metadata[i] for i in indices[0] if i != -1]

def list_docs() -> List[str]:
    return list({m["source"] for m in _metadata})

def delete_doc(filename: str) -> bool:
    global _index, _metadata
    new_meta = [m for m in _metadata if m["source"] != filename]
    if len(new_meta) == len(_metadata):
        return False
    _metadata = new_meta
    _index = faiss.IndexFlatL2(DIM)
    if new_meta:
        _index.add(_embed([m["text"] for m in new_meta]))
    _save_store()
    return True

_load_store()

# --- RAG logic - retrieve chunks then ask groq ---

# tried a few prompts, this one works best for staying on topic
SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based strictly on the provided document context.
- Answer ONLY from the context provided.
- If the context lacks enough information, say: "I couldn't find relevant information in the uploaded documents."
- Be concise and accurate.
- Cite the source filename when referencing specific information.
- Use the conversation history to handle follow-up questions."""

def rag_answer(question: str, history: List[Dict[str, str]]) -> Tuple[str, List[str]]:
    if not GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY is not set.")
    chunks = search_store(question)
    if not chunks:
        return "No documents have been uploaded yet. Please upload a PDF, DOCX, or TXT file first.", []

    sources = list({c["source"] for c in chunks})
    context = "\n\n---\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": t["role"], "content": t["content"]} for t in history[-10:]]
    messages.append({"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {question}"})

    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=GROQ_MODEL, messages=messages,
        max_tokens=MAX_TOKENS, temperature=0.2,
    )
    return resp.choices[0].message.content.strip(), sources

# --- fastapi app setup ---
app = FastAPI(title="DocuMind RAG Chatbot")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# keep chat history per session in memory
chat_histories: Dict[str, List[Dict[str, str]]] = {}

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str

class UploadResponse(BaseModel):
    filename: str
    chunks: int
    message: str

# serve the frontend from the same server
@app.get("/", response_class=HTMLResponse)
async def ui():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".pdf", ".docx", ".txt"):
        raise HTTPException(400, f"Unsupported type '{ext}'. Use .pdf, .docx, or .txt")
    content = await file.read()
    try:
        chunks = process_document(content, ext, filename)
        n = add_to_store(chunks)
        return UploadResponse(filename=filename, chunks=n,
                              message=f"✓ Processed '{filename}' → {n} chunks indexed.")
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    sid = req.session_id or "default"
    history = chat_histories.setdefault(sid, [])
    try:
        answer, sources = rag_answer(req.question, history)
    except Exception as e:
        raise HTTPException(500, str(e))
    history.append({"role": "user",      "content": req.question})
    history.append({"role": "assistant", "content": answer})
    chat_histories[sid] = history[-20:]
    return ChatResponse(answer=answer, sources=sources, session_id=sid)

@app.delete("/chat/{session_id}")
async def clear_chat(session_id: str):
    chat_histories.pop(session_id, None)
    return {"message": f"History cleared for '{session_id}'."}

@app.get("/documents")
async def documents():
    return {"documents": list_docs()}

@app.delete("/documents/{filename}")
async def remove_document(filename: str):
    if not delete_doc(filename):
        raise HTTPException(404, "Document not found.")
    return {"message": f"'{filename}' removed."}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)