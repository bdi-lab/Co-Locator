from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
import uvicorn
from typing import List

app = FastAPI()

model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True).to(torch.bfloat16)
model._first_module().auto_model.config.use_cache = False
model.max_seq_length = 4096
model.to("cuda")

class EmbedRequest(BaseModel):
    sentences: List[str]

@app.post("/embed")
async def get_embeddings(request: EmbedRequest):
    try:
        embeddings = model.encode(
            request.sentences, 
            batch_size=4, 
            normalize_embeddings=True, 
            convert_to_numpy=True
        )
        return {"embeddings": embeddings.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)