import os
import json
import pandas as pd
import math 
import psycopg2
from fastapi import FastAPI
from pydantic import BaseModel
from tqdm import tqdm
from openai import OpenAI
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI application
app = FastAPI(title="Excel Q&A API")

# vLLM client (OpenAI-compatible)
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")
client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY,
)

# Helpers for Excel processing and text chunking
def excel_to_text(path: str) -> str:
    """
    Read an Excel file and convert each row to a pipe-separated line.
    Returns a multiline string containing the entire file.
    """
    try:
        df = pd.read_excel(path)
        lines = []
        for _, row in df.iterrows():
             fields = [str(x) for x in row if pd.notna(x)]
             if fields:
                 lines.append(" | ".join(fields))
        return "\n".join(lines)
        ...
        except Exception as e:
            logger.error(f"Failed to process {path}: {e}")
            return ""
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """
    Split text into overlapping word chunks.
    """
    words = text.split()
    if not words:
        return
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(words), step):
        yield " ".join(words[i:i + chunk_size])

# Request model for the /ask endpoint
class QueryRequest(BaseModel):
    query: str

# Database setup (PostgreSQL + pgvector)
DB_DSN = os.getenv("DATABASE_URL")
if not DB_DSN:
    raise RuntimeError("DATABASE_URL not set (expected a PostgreSQL connection string).")
conn = psycopg2.connect(DB_DSN)
conn.autocommit = True
def init_db() -> None:
    # Create pgvector extension, table, and index if they don't exist.
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS excel_docs (
                id TEXT PRIMARY KEY,
                document TEXT,
                embedding vector(768),
                source TEXT
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_excel_docs_embedding
            ON excel_docs
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
@app.on_event("startup")
def on_startup():
    init_db()

# Embedding helpers
def to_vector_literal(embedding) -> str:
    # Convert a list of floats to a pgvector string literal.
    return "[" + ",".join(str(x) for x in embedding) + "]"

def embed_literal(text: str) -> str:
    # Get an embedding for the given text using vLLM and convert it into a pgvector literal string.
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    embedding = response.data[0].embedding
    return to_vector_literal(embedding)

# System prompt for the LLM
SYSTEM_PROMPT = """
You are a sales assistant that answers questions using ONLY the provided Excel context.
The Excel files contain quotes, pricing, SKUs, terms, and CRM-like data.
Your goal is to help sales reps move from quote -> signed contract.
RULES:
- Use ONLY the given context. If it's not there, say you don't know.
- Never invent prices, discounts, legal terms, or customer details.
- Maintain a professional, friendly, consultative tone suitable for B2B sales.
- Do not answer or engage with disallowed topics (hate, self-harm, sexual content, extremism, etc.).
CONTEXT HANDLING:
- If the context is empty or clearly not relevant, you MUST:
  - Set "json.enough_context": false
  - Explicitly state in both "markdown" and "json.answer" that you do not have enough information to answer.
- If the context is sufficient and relevant, set "json.enough_context": true.
You MUST always respond as a VALID JSON object with this exact schema:
{
  "markdown": "<short, clear answer in Markdown using only the context>",
  "json": {
    "answer": "<same short answer as plain text>",
    "enough_context": true/false,
    "confidence": <number between 0 and 1>,

    "tone": {
      "style": "consultative_sales",
      "polite": true/false,
      "issues": []
    },

    "policy": {
      "allowed": true/false,
      "category": "<'safe' | 'disallowed_topic'>",
      "reason": "<short explanation>"
    },

    "sales": {
      "next_best_action": "<concrete suggestion for the sales rep>",
      "follow_up_prompt": "<email/message snippet>"
    },

    "retrieval": {
      "best_distance": <number or null>
    }
  }
}
"""

# Endpoint: ingest Excel files into PostgreSQL
@app.post("/ingest")
def ingest_excels():
    # Read all Excel files in 'excel_data', convert them to text, split into chunks, embed each chunk, and store in PostgreSQL.
    logger.info("Starting ingestion from 'excel_data' folder")
    added = 0
    for filename in os.listdir("excel_data"):
        if not filename.lower().endswith((".xls", ".xlsx")):
            continue
        path = os.path.join("excel_data", filename)
        text = excel_to_text(path)
        if not text.strip():
            continue
        for i, chunk in enumerate(chunk_text(text)):
            embedding_literal = embed_literal(chunk)
            doc_id = f"{filename}_{i}"
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO excel_docs (id, document, embedding, source)
                    VALUES (%s, %s, %s::vector, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        document = EXCLUDED.document,
                        embedding = EXCLUDED.embedding,
                        source = EXCLUDED.source;
                """, (doc_id, chunk, embedding_literal, filename))
            added += 1
    logger.info("Ingestion completed. Added %d chunks", added)
    return {"status": "ok", "message": f"Embedded and stored {added} chunks."}

# Endpoint: ask a question using Excel-based context (RAG)
@app.post("/ask")
def ask_excel(request: QueryRequest):
    logger.info("New query received: %s", request.query)
    # Embed the user query, retrieve top-3 most relevant Excel chunks, send them to the LLM, and return a structured sales-ready JSON answer.
    query_embedding_literal = embed_literal(request.query)
    # Retrieve id, document, source and vector distance
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, document, source, embedding <#> %s::vector AS distance
            FROM excel_docs
            ORDER BY distance
            LIMIT 3;
        """, (query_embedding_literal,))
        rows = cur.fetchall()
        logger.info("Retrieved %d chunks from DB", len(rows or []))
    # Build plain text context (documents only) + track best_distance
    documents = []
    best_distance = None
    for idx, (doc_id, document, source, distance) in enumerate(rows or []):
        distance = float(distance)
        if idx == 0:
            best_distance = distance
        documents.append(document)
    context = "\n\n".join(documents) if documents else ""
    # Send to LLM
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.query}"}
        ]
    )
    raw = response.choices[0].message.content or ""
    # Try parsing JSON response
    try:
        parsed = json.loads(raw)
        answer_markdown = (parsed.get("markdown") or "").strip()
        answer_json = parsed.get("json") or {}
    except Exception:
        answer_markdown = raw.strip()
        answer_json = {}
    # Include plain-text answer
    answer_json.setdefault("answer", answer_markdown)
    # enough_context basato sia sulla presenza di contesto sia sulla distanza
    has_context = bool(context.strip())
    if best_distance is not None:
        enough_context = has_context and best_distance < 0.25  # soglia regolabile
    else:
        enough_context = has_context
    answer_json.setdefault("enough_context", enough_context)
    # Ensure sales fields exist
    sales = answer_json.get("sales") or {}
    sales.setdefault("next_best_action", "")
    sales.setdefault("follow_up_prompt", "")
    answer_json["sales"] = sales
    # Ensure tone exists
    tone = answer_json.get("tone") or {}
    tone.setdefault("style", "consultative_sales")
    tone.setdefault("polite", True)
    tone.setdefault("issues", [])
    answer_json["tone"] = tone
    # Ensure policy exists (minimum moderation)
    policy = answer_json.get("policy") or {}
    policy.setdefault("allowed", True)
    policy.setdefault("category", "safe")
    answer_json["policy"] = policy
    # Include best_distance for diagnostics/UI
    retrieval = answer_json.get("retrieval") or {}
    retrieval.setdefault("best_distance", best_distance)
    answer_json["retrieval"] = retrieval
    # Block disallowed topics
    if not policy.get("allowed", True):
        return {
            "query": request.query,
            "blocked": True,
            "policy_category": policy.get("category", "disallowed_topic"),
            "answer_markdown": answer_markdown,
            "answer_json": answer_json,
        }
    # NOTE for UI:
    # - Render `answer_markdown` in Markdown.
    # - Show sales.next_best_action as “Next best action”.
    # - Show sales.follow_up_prompt as a pre-filled suggested message.
    return {
        "query": request.query,
        "answer_markdown": answer_markdown,
        "answer_json": answer_json
    }
