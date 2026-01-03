from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer
import os

app = FastAPI()

# 경량 CPU 임베딩 모델
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

@app.post("/embed")
async def embed_text(req: Request):
    body = await req.json()
    advice_text = body.get("advice_text", "")
    
    if not advice_text:
        return {"error": "advice_text is required"}
    
    embedding_vector = model.encode(advice_text).tolist()
    return {"embedding": embedding_vector}

@app.get("/")
async def root():
    return {"status": "ok"}
