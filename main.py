import os
import json
import logging
from typing import List
import pandas as pd
import psycopg2
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI application
app = FastAPI(title="AT Mind Sales Q&A API")

# vLLM / Ollama-compatible clients (OpenAI-compatible)
VLLM_CHAT_BASE_URL = os.getenv("VLLM_CHAT_BASE_URL", "http://localhost:8001/v1")
VLLM_EMBED_BASE_URL = os.getenv("VLLM_EMBED_BASE_URL", "http://localhost:8002/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "token-abc123")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama-3.2-3b-instruct")

chat_client = OpenAI(base_url=VLLM_CHAT_BASE_URL, api_key=VLLM_API_KEY)
embed_client = OpenAI(base_url=VLLM_EMBED_BASE_URL, api_key=VLLM_API_KEY)

# Database setup (PostgreSQL + pgvector)
DB_DSN = os.getenv("DATABASE_URL")
if not DB_DSN:
    raise RuntimeError("DATABASE_URL is required for Postgres + pgvector backend.")

conn = None

def get_db_conn():
    global conn
    if conn is None or getattr(conn, "closed", 0) != 0:
        conn = psycopg2.connect(DB_DSN)
        conn.autocommit = True
    return conn

def init_db() -> None:
    connection = get_db_conn()
    with connection.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sales_docs (
                id TEXT PRIMARY KEY,
                document TEXT,
                embedding vector(768),
                source TEXT
            );
        """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sales_docs_embedding
            ON sales_docs
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """
        )

@app.on_event("startup")
def on_startup():
    init_db()
    logger.info("Startup complete.")

# --------- Helpers for Excel processing and chunking ---------
def excel_to_text(path: str) -> str:
    """
    Read an Excel file and convert each row to a pipe-separated line. Returns a multiline string containing the entire file.
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

# --------- Embedding helpers ---------
def get_embedding(text: str) -> List[float]:
    resp = embed_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return resp.data[0].embedding

def to_vector_literal(embedding: List[float]) -> str:
    return "[" + ",".join(str(x) for x in embedding) + "]"

# --------- Request model ---------
class QueryRequest(BaseModel):
    query: str
# --------- System prompt / JSON schema requirements ---------

SYSTEM_PROMPT = """
You are a sales assistant that answers questions using ONLY the provided context.
The files contain quotes, pricing, SKUs, terms, and CRM-like data from Autotorino.
Your goal is to help sales reps move from quote -> signed contract.

RULES:
- Always answer in English unless the user explicitly writes the question in another language.
- Use ONLY the given context. If it's not there, say you don't know.
- Never invent prices, discounts, legal terms, or customer details.
- Maintain a professional, friendly, consultative tone suitable for B2B sales.
- Do not answer or engage with disallowed topics (hate, self-harm, sexual content, extremism, etc.).
- Questions about customers, prices, discounts, quotes, SKUs, contracts, and sales negotiations are ALWAYS allowed and must NOT be treated as disallowed topics.
- ALWAYS answer in English, even if the user writes in another language.
- All output fields ("markdown", "json.answer", "sales.next_best_action", "sales.follow_up_prompt") MUST be in clear, fluent English only.
- Do NOT use Italian or any other language under any circumstances.

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
    return {"status": "ok", "backend": "postgres_pgvector"}

# --------- Endpoint: ingest Excel files into PostgreSQL ---------
@app.post("/ingest")
def ingest_excels():
    """
    Read all Excel files in 'excel_data', convert to text, chunk, embed, and store in Postgres (pgvector).
    """
    folder = "excel_data"
    if not os.path.isdir(folder):
        msg = f"Folder '{folder}' not found."
        logger.error(msg)
        return {"status": "error", "message": msg}

    logger.info("Starting ingestion from 'excel_data' folder")
    added = 0
    file_list = [f for f in os.listdir(folder) if f.lower().endswith((".xlsx", ".xls"))]

    connection = get_db_conn()

    for filename in file_list:
        full_path = os.path.join(folder, filename)
        text = excel_to_text(full_path)
        if not text.strip():
            logger.warning("No text extracted from %s, skipping", filename)
            continue

        for i, chunk in enumerate(chunk_text(text)):
            try:
                emb = get_embedding(chunk)
                embedding_literal = to_vector_literal(emb)
                doc_id = f"{filename}_{i}"
                with connection.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO sales_docs (id, document, embedding, source)
                        VALUES (%s, %s, %s::vector, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            document = EXCLUDED.document,
                            embedding = EXCLUDED.embedding,
                            source = EXCLUDED.source;
                        """,
                        (doc_id, chunk, embedding_literal, filename),
                    )
                added += 1
            except Exception as e:
                logger.error("Error embedding/storing chunk %s_%d: %s", filename, i, e)

    logger.info("Ingestion completed. Added %d chunks", added)
    return {"status": "ok", "message": f"Embedded and stored {added} chunks."}

# --------- Endpoint: ask a question using RAG ---------
@app.post("/ask")
def ask_sales(request: QueryRequest):
    logger.info("New query received: %s", request.query)

    connection = get_db_conn()

    # 1) Embed the query and retrieve top-k chunks
    query_embedding = get_embedding(request.query)
    query_embedding_literal = to_vector_literal(query_embedding)

    with connection.cursor() as cur:
        cur.execute(
            """
            SELECT id, document, source, embedding <#> %s::vector AS distance
            FROM sales_docs
            ORDER BY distance
            LIMIT 3;
            """,
            (query_embedding_literal,),
        )
        rows = cur.fetchall()

    documents = []
    best_distance = None
    for idx, (doc_id, document, source, distance) in enumerate(rows or []):
        distance = float(distance)
        if idx == 0:
            best_distance = distance
        documents.append(document)

    context = "\n\n".join(documents) if documents else ""
    has_context = bool(context.strip())

    # 2) Call the chat model with system prompt + context
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

    # 3) Parse model JSON, enforcing schema and guardrails
    try:
        parsed = json.loads(raw)
        answer_markdown = (parsed.get("markdown") or "").strip()
        answer_json = parsed.get("json") or {}
    except Exception:
        # If the model didn't return valid JSON, fallback to minimal structure
        answer_markdown = raw.strip()
        answer_json = {}

    # Ensure plain-text answer
    if not answer_json.get("answer"):
        answer_json["answer"] = answer_markdown

    # Determine enough_context from retrieval
    if best_distance is not None:
        enough_context = has_context and best_distance < 0.25  # adjustable threshold
    else:
        enough_context = has_context

    answer_json.setdefault("enough_context", enough_context)
    # If we don't have enough context, force a safe fallback answer
    if not answer_json.get("enough_context", False):
        fallback_msg = (
            "I don’t have enough relevant information in the current knowledge base "
            "to answer this question reliably."
        )
        answer_markdown = fallback_msg
        answer_json["answer"] = fallback_msg
        answer_json["enough_context"] = False
        # Keep confidence low when we lack context
        existing_conf = answer_json.get("confidence", 0.2)
        answer_json["confidence"] = min(existing_conf, 0.3)


    # Confidence heuristic (simple, based on context availability)
    if "confidence" not in answer_json:
        answer_json["confidence"] = 1.0 if enough_context else 0.2

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

    # Ensure policy exists (record metadata but do NOT block normal sales queries)
    policy = answer_json.get("policy") or {}
    policy.setdefault("allowed", True)
    policy.setdefault("category", "safe")
    policy.setdefault("reason", "no issues detected")
    answer_json["policy"] = policy

    # Normalize policy: for this sales assistant, only block if the model explicitly flags
    # 'disallowed_topic'. All sales-related questions are allowed.
    if policy.get("category") != "disallowed_topic":
        policy["allowed"] = True

    # Ensure retrieval info exists
    retrieval = answer_json.get("retrieval") or {}
    retrieval["best_distance"] = best_distance
    answer_json["retrieval"] = retrieval

    # Clean up weird templating / JSON artifacts the model might return
    # We want markdown and json.answer to be plain natural-language text.
    if "{{" in answer_markdown and "}}" in answer_markdown:
        lines = [l.strip() for l in answer_markdown.splitlines() if l.strip()]
        # Pick the first line that looks like real text (not {, }, ", {{, or json)
        for line in lines:
            if not (
                line.startswith("{")
                or line.startswith("}")
                or line.startswith('"')
                or line.startswith("{{")
                or line.lower().startswith("json")
            ):
                answer_markdown = line
                break
        answer_json["answer"] = answer_markdown

    # 4) Final response STRICTLY matches required schema
    return {
        "markdown": answer_markdown,
        "json": answer_json,
    }
