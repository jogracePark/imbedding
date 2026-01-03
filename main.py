from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer

app = FastAPI()

# --------------------
# 초경량 CPU 임베딩 모델
# --------------------
# MiniLM L3 계열: ~20MB, 512MB RAM 안전
model = SentenceTransformer("sentence-transformers/all-MiniLM-L3-v2")

@app.post("/embed")
async def embed_text(req: Request):
    body = await req.json()
    advice_text = body.get("advice_text", "")
    
    if not advice_text:
        return {"error": "advice_text is required"}
    
    # 한 문장씩 encode → 메모리 절약
    embedding_vector = model.encode([advice_text])[0].tolist()
    return {"embedding": embedding_vector}

@app.get("/")
async def root():
    return {"status": "ok"}
