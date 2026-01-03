from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="MiniLM-L6 Embedding API")

# ------------------------
# 모델 로딩 (Render 무료 tier에서 CPU만으로 가능)
# ------------------------
print("모델 로딩 중...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("모델 로딩 완료!")

# ------------------------
# 요청 형식
# ------------------------
class EmbeddingRequest(BaseModel):
    text: str

# ------------------------
# 루트 헬스체크
# ------------------------
@app.get("/")
def root():
    return {"status": "ok"}

# ------------------------
# 임베딩 생성 엔드포인트
# ------------------------
@app.post("/embed")
def embed(req: EmbeddingRequest):
    try:
        embedding = model.encode(req.text, normalize_embeddings=True)
        return {"embedding": embedding.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
