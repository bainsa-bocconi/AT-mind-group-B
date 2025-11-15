import os
import ollama
import chromadb
import pandas as pd
from tqdm import tqdm

#setup and functions
client=chromadb.client()
collection=client.get_or_create_collection("excel_docs")

def excel_to_text(path) -> str:
    try:
        df=pd.read_excel(path)
        text - ""
        for _, row in df.iterrows():
            #combine to one string
            text+= " | ".join(str(x) for x in row if pd.notna(x)) + "\n"
        return text
    except Exception as e:
        print(f"Fail {path}: {e}")
        return ""
    
def chunck_text(text, chunk_size=500, overlap=50):
    words=text.split()
    for i in range(0, len(words), chunk_size - overlap):
        yield " ".join(words[i:i+chunk_size])