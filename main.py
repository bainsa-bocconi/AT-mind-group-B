import os
import json
import pandas as pd
import psycopg2
from fastapi import FastAPI
from pydantic import BaseModel
from tqdm import tqdm
from openai import OpenAI


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
    #Read an Excel file and convert each row to a pipe-separated line. Returns a multiline string containing the entire file.
    try:
        df = pd.read_excel(path)
        text = ""
        for _, row in df.iterrows():
            line = " | ".join(str(x) for x in row if pd.notna(x))
            text += line + "\n"
        return text
    except Exception as e:
        print(f"Failed to process {path}: {e}")
        return ""
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """
    Split text into overlapping word chunks.
    """
    words = text.split()
    step = chunk_size - overlap
    if step <= 0:
        step = 1

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
    """Create pgvector extension, table, and index if they don't exist."""
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
    """
    Convert a list of floats to a pgvector string literal.
    Example: [0.1,0.2,0.3]
    """
    return "[" + ",".join(str(x) for x in embedding) + "]"
def embed_literal(text: str) -> str:
    """
    Get an embedding for the given text using vLLM and convert it
    into a pgvector literal string.
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    embedding = response.data[0].embedding
    return to_vector_literal(embedding)

# System prompt for the LLM
SYSTEM_PROMPT = """
You are an assistant that answers questions using ONLY the provided Excel context.
If the context is not enough to answer, say that clearly.
Keep answers short and clear.

Always respond as a VALID JSON object with this exact schema:
{
  "markdown": "<short, clear answer in Markdown using only the context>",
  "json": {
    "answer": "<same short answer as plain text>",
    "enough_context": true/false
  }
}
Do not include any other top-level fields. Do not wrap in code fences.
"""

# Endpoint: ingest Excel files into PostgreSQL
@app.post("/ingest")
def ingest_excels():
    """
    Read all Excel files in 'excel_data', convert them to text,
    split into chunks, embed each chunk, and store in PostgreSQL.
    """
    added = 0

    for filename in tqdm(os.listdir("excel_data")):
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

    return {"status": "ok", "message": f"Embedded and stored {added} chunks."}

# Endpoint: ask a question using Excel-based context (RAG)
@app.post("/ask")
def ask_excel(request: QueryRequest):
    """
    Embed the user query, retrieve the most relevant Excel text chunks,
    send them to the LLM, and return the structured JSON answer.
    """
    query_embedding_literal = embed_literal(request.query)

    with conn.cursor() as cur:
        cur.execute("""
            SELECT document
            FROM excel_docs
            ORDER BY embedding <#> %s::vector
            LIMIT 3;
        """, (query_embedding_literal,))
        rows = cur.fetchall()

    context = "\n\n".join(row[0] for row in rows) if rows else ""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.query}"}
        ]
    )

    raw = response.choices[0].message.content or ""

    # Try to parse as JSON
    try:
        parsed = json.loads(raw)
        answer_markdown = parsed.get("markdown", "").strip()
        answer_json = parsed.get("json") or {}
    except Exception:
        answer_markdown = raw.strip()
        answer_json = {}

    answer_json.setdefault("answer", answer_markdown)
    answer_json.setdefault("enough_context", bool(context.strip()))

    return {
        "query": request.query,
        "answer_markdown": answer_markdown,
        "answer_json": answer_json
    }
