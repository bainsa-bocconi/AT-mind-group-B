import os
import ollama
import chromadb
import pandas as pd
from tqdm import tqdm

#Setup
client=chromadb.Client()
collection=client.get_or_create_collection("excel_docs")

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
        
#Process all Excel files and store embeddings       
for filename in tqdm(os.listdir("excel_data")):
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        path=os.path.join("excel_data", filename)
        text=excel_to_text(path)
        if not text.strip():
            continue
        for i, chunk in enumerate(chunk_text(text)):
            emb=ollama.embeddings(model="mxbai-embed-large", prompt=chunk)["embeddings"]
            collection.add(
                documents=[chunk],
                ids=[f"{filename}_{i}"],
                embeddings=[emb],
                metadatas=[{"source": filename}],
            )
print("Embedded and stored.")

#Qyuery&answer
query= "Summarize the company's general customer satisfaction"
q_emb=ollama.embeddings(model="mxbai-embed-large", prompt=query)["embedding"]

results=collection.query(query_embeddings=[q_emb], n_results=3)
context= "\n\n".join(results["documents"][0])

response=ollama.chat(
    model="llama3",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that uses the provided Excel context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
)

print("\n---Answer---\n")
print(response["message"]["content"])