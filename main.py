import os
import ollama
import chromadb
import json
import pandas as pd
from tqdm import tqdm
from fastapi import FastAPI
from pydantic import BaseModel

#Setup
client=chromadb.Client()
collection=client.get_or_create_collection("excel_docs")

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

app = FastAPI(title="Excel Q&A API")

#Helpers
def excel_to_text(path) -> str:
    #Read an Excel file and convert each orow to a pip-seoarated line
    try:
        df=pd.read_excel(path)
        text = ""
        for _, row in df.iterrows():
            #combine to one string
            text+= " | ".join(str(x) for x in row if pd.notna(x)) + "\n"
        return text
    except Exception as e:
        print(f"Fail {path}: {e}")
        return ""
    
def chunk_text(text, chunk_size=500, overlap=50):
    words=text.split()
    for i in range(0, len(words), chunk_size - overlap):
        yield " ".join(words[i:i+chunk_size])

# Request model
class QueryRequest(BaseModel):
    query: str
    
# Endpoint to ingest Excel files
@app.post("/ingest")
def ingest_excels():
    added = 0
    for filename in tqdm(os.listdir("excel_data")):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            path = os.path.join("excel_data", filename)
            text = excel_to_text(path)
            if not text.strip():
                continue
            for i, chunk in enumerate(chunk_text(text)):
                emb = ollama.embeddings(model="mxbai-embed-large", prompt=chunk)["embedding"]
                collection.add(
                    documents=[chunk],
                    ids=[f"{filename}_{i}"],
                    embeddings=[emb],
                    metadatas=[{"source": filename}],
                )
                added += 1
    return {"status": "ok", "message": f"Embedded and stored {added} chunks."}

# Endpoint to ask a question
@app.post("/ask")
def ask_excel(request: QueryRequest):
    query = request.query
    q_emb = ollama.embeddings(model="mxbai-embed-large", prompt=query)["embedding"]
    results = collection.query(query_embeddings=[q_emb], n_results=3)

    # Handle empty results safely
    docs = results.get("documents", [[]])
  
    top_docs = docs[0] if docs and len(docs) > 0 else []
    context = "\n\n".join(top_docs) if top_docs else ""

    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )

    content = response["message"]["content"]

    # Try to parse model output as JSON per schema; fallback if needed 
    answer_markdown = ""
    answer_json = {}
    try:
        obj = json.loads(content)
        answer_markdown = obj.get("markdown", "").strip()
        answer_json = obj.get("json", {})
        if "answer" not in answer_json:
            answer_json["answer"] = answer_markdown
        if "enough_context" not in answer_json:
            answer_json["enough_context"] = bool(context.strip())
    except Exception:
        # Fallback: construct both from raw text
        answer_markdown = content.strip() if content else ""
        answer_json = {
            "answer": answer_markdown,
            "enough_context": bool(context.strip())
        }

    return {
        "query": query,
        "answer_markdown": answer_markdown,  
        "answer_json": answer_json,           
    }
