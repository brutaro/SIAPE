from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from openai import OpenAI

# Carrega .env
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "SIAPE"

# Inicializa clientes
app = FastAPI()
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Modelo da requisição
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

# Endpoint de busca
@app.post("/search")
def search(request: QueryRequest):
    try:
        # Embedding da pergunta
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[request.question]
        )
        vector = response.data[0].embedding

        # Busca no Qdrant
        result = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=request.top_k
        )

        # Retorna os textos encontrados
        return {
            "question": request.question,
            "results": [
                {
                    "text": hit.payload["text"],
                    "score": hit.score
                } for hit in result
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))