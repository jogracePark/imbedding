from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer
import os

app = FastAPI()

model_path = os.path.join(os.path.dirname(__file__), "local_model")
model = SentenceTransformer(model_path)  # 로컬 모델 로드

@app.post("/embed")
async def embed_text(req: Request):
    body = await req.json()
    advice_text = body.get("advice_text", "")
    if not advice_text:
        return {"error": "advice_text is required"}
    embedding_vector = model.encode([advice_text])[0].tolist()
    return {"embedding": embedding_vector}

@app.get("/")
async def root():
    return {"status": "ok"}
