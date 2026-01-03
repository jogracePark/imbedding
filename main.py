from fastapi import FastAPI, Request
from transformers import pipeline
import torch
import os

app = FastAPI()

# --------------------
# 로컬 임베딩 모델 설정
# --------------------
EMBEDDING_MODEL = "Xenova/paraphrase-multilingual-MiniLM-L12-v2"

# 임베딩 파이프라인 초기화 (CPU)
embedder = pipeline("feature-extraction", model=EMBEDDING_MODEL, device=-1)

@app.post("/embed")
async def embed_text(req: Request):
    body = await req.json()
    advice_text = body.get("advice_text", "")
    
    if not advice_text:
        return {"error": "advice_text is required"}
    
    # 문장 임베딩 생성
    output = embedder(advice_text, pooling="mean", normalize=True)
    embedding_vector = output[0]  # 첫 번째 문장
    return {"embedding": embedding_vector}

@app.get("/")
async def root():
    return {"status": "ok"}
