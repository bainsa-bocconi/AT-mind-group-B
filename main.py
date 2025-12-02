import os
import json
import math
import logging
from typing import List, Dict, Any

import pandas as pd
import psycopg2
from fastapi import FastAPI
from pydantic import BaseModel
from tqdm import tqdm  # optional but kept for potential progress bars
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI application
app = FastAPI(title="Excel Q&A API")

# vLLM / Ollama-compatible client (OpenAI-compatible)
# IMPORTANT: make sure something is running at this URL that supports
# /embeddings and /chat/completions (OpenAI-compatible API)
VLLM_CHAT_BASE_URL = os.getenv("VLLM_CHAT_BASE_URL", "http://localhost:8001/v1")
VLLM_EMBED_BASE_URL = os.getenv("VLLM_EMBED_BASE_URL", "http://localhost:8002/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "token-abc123")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
CHAT_MODEL = os.getenv("CHAT_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

chat_client = OpenAI(
    base_url=VLLM_CHAT_BASE_URL,
    api_key=VLLM_API_KEY,
)

embed_client = OpenAI(
    base_url=VLLM_EMBED_BASE_URL,
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
    except Exception as e:
        logger.error(f"Failed to process {path}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 20):
    """
    Split text into overlapping word chunks.
    """
    words = text.split()
    if not words:
        return
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(words), step):
        yield " ".join(words[i : i + chunk_size])


# Request model for the /ask endpoint
class QueryRequest(BaseModel):
    query: str


# Database setup (PostgreSQL + pgvector)
DB_DSN = os.getenv("DATABASE_URL")
USE_DB = bool(DB_DSN)
conn = None

if USE_DB:
    logger.info("DATABASE_URL found, enabling Postgres + pgvector backend.")
else:
    logger.warning("DATABASE_URL not set. Using IN-MEMORY store only (no persistence).")

INMEMORY_DOCS: List[Dict[str, Any]] = []


def get_db_conn():
    global conn
    if not USE_DB:
        return None
    if conn is None or getattr(conn, "closed", 0) != 0:
        conn = psycopg2.connect(DB_DSN)
        conn.autocommit = True
    return conn


def init_db() -> None:
    if not USE_DB:
        return
    connection = get_db_conn()
    with connection.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS excel_docs (
                id TEXT PRIMARY KEY,
                document TEXT,
                embedding vector(768),
                source TEXT
            );
        """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_excel_docs_embedding
            ON excel_docs
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """
        )


@app.on_event("startup")
def on_startup():
    init_db()
    logger.info("Startup complete.")


# Embedding helpers
def get_embedding(text: str) -> List[float]:
    resp = embed_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return resp.data[0].embedding


def to_vector_literal(embedding: List[float]) -> str:
    return "[" + ",".join(str(x) for x in embedding) + "]"


def embed_literal(text: str) -> str:
    emb = get_embedding(text)
    return to_vector_literal(emb)


def cosine_distance(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    cos_sim = dot / (norm_a * norm_b)
    return 1.0 - cos_sim


# System prompt for the LLM
SYSTEM_PROMPT = """
You are a sales assistant that answers questions using ONLY the provided Excel context.
The Excel files contain quotes, pricing, SKUs, terms, and CRM-like data.
Your goal is to help sales reps move from quote -> signed contract.
RULES:
-Always answer in English unless the user explicitly writes the question in another language.
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


@app.get("/")
def health():
    return {"status": "ok", "backend": "postgres" if USE_DB else "in_memory"}


# Endpoint: ingest Excel files into PostgreSQL or in-memory
@app.post("/ingest")
def ingest_excels():
    """
    Read all Excel files in 'excel_data', convert to text, chunk, embed,
    and store either in Postgres (if configured) or in-memory.
    """
    folder = "excel_data"
    if not os.path.isdir(folder):
        msg = f"Folder '{folder}' not found."
        logger.error(msg)
        return {"status": "error", "message": msg}

    logger.info("Starting ingestion from 'excel_data' folder")
    added = 0

    # Optionally use tqdm for progress if many files
    file_list = [f for f in os.listdir(folder) if f.lower().endswith((".xlsx", ".xls"))]

    for filename in tqdm(file_list, desc="Ingesting Excel files"):
        full_path = os.path.join(folder, filename)
        text = excel_to_text(full_path)
        if not text.strip():
            logger.warning("No text extracted from %s, skipping", filename)
            continue

        for i, chunk in enumerate(chunk_text(text)):
            try:
                if USE_DB:
                    embedding_literal = embed_literal(chunk)
                    doc_id = f"{filename}_{i}"
                    connection = get_db_conn()
                    with connection.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO excel_docs (id, document, embedding, source)
                            VALUES (%s, %s, %s::vector, %s)
                            ON CONFLICT (id) DO UPDATE SET
                                document = EXCLUDED.document,
                                embedding = EXCLUDED.embedding,
                                source = EXCLUDED.source;
                            """,
                            (doc_id, chunk, embedding_literal, filename),
                        )
                else:
                    emb = get_embedding(chunk)
                    INMEMORY_DOCS.append(
                        {
                            "id": f"{filename}_{i}",
                            "document": chunk,
                            "source": filename,
                            "embedding": emb,
                        }
                    )
                added += 1
            except Exception as e:
                logger.error("Error embedding/storing chunk %s_%d: %s", filename, i, e)

    logger.info("Ingestion completed. Added %d chunks", added)
    return {"status": "ok", "message": f"Embedded and stored {added} chunks."}


# Endpoint: ask a question using Excel-based context (RAG)
@app.post("/ask")
def ask_excel(request: QueryRequest):
    logger.info("New query received: %s", request.query)

    documents: List[str] = []
    best_distance = None

    if USE_DB:
        # Use Postgres + pgvector
        query_embedding_literal = embed_literal(request.query)
        connection = get_db_conn()
        with connection.cursor() as cur:
            cur.execute(
                """
                SELECT id, document, source, embedding <#> %s::vector AS distance
                FROM excel_docs
                ORDER BY distance
                LIMIT 3;
                """,
                (query_embedding_literal,),
            )
            rows = cur.fetchall()
            logger.info("Retrieved %d chunks from DB", len(rows or []))

        for idx, (doc_id, document, source, distance) in enumerate(rows or []):
            distance = float(distance)
            if idx == 0:
                best_distance = distance
            documents.append(document)
    else:
        # Use in-memory store
        if not INMEMORY_DOCS:
            return {
                "status": "error",
                "message": "No documents in memory. Please call /ingest first.",
            }

        query_emb = get_embedding(request.query)
        scored = []
        for doc in INMEMORY_DOCS:
            dist = cosine_distance(query_emb, doc["embedding"])
            scored.append((dist, doc["document"], doc["source"]))

        scored.sort(key=lambda x: x[0])
        top3 = scored[:3]
        for idx, (dist, document, source) in enumerate(top3):
            if idx == 0:
                best_distance = dist
            documents.append(document)

        logger.info("Retrieved %d chunks from in-memory store", len(top3))

    context = "\n\n".join(documents) if documents else ""

    # Send to LLM
    response = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {request.query}",
            },
        ],
    )
    raw = response.choices[0].message.content or ""

    # Try parsing JSON response from the model
    try:
        parsed = json.loads(raw)
        answer_markdown = (parsed.get("markdown") or "").strip()
        answer_json = parsed.get("json") or {}
    except Exception:
        # If the model didn't return valid JSON, fall back to a simple structure
        answer_markdown = raw.strip()
        answer_json = {}

    # Include plain-text answer
    answer_json.setdefault("answer", answer_markdown)

    # enough_context based on presence of context and distance
    has_context = bool(context.strip())
    if best_distance is not None:
        enough_context = has_context and best_distance < 0.25  # adjustable threshold
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
    policy.setdefault("reason", "no issues detected")
    answer_json["policy"] = policy

    # Include best_distance for diagnostics/UI
    retrieval = answer_json.get("retrieval") or {}
    retrieval.setdefault("best_distance", best_distance)
    answer_json["retrieval"] = retrieval

    # Block disallowed topics if the model flagged them
    if not policy.get("allowed", True):
        return {
            "query": request.query,
            "blocked": True,
            "policy_category": policy.get("category", "disallowed_topic"),
            "answer_markdown": answer_markdown,
            "answer_json": answer_json,
        }

    # Normal successful answer
    return {
        "query": request.query,
        "answer_markdown": answer_markdown,
        "answer_json": answer_json,
    }
