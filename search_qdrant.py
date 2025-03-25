import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from openai import OpenAI

# Carrega variáveis do .env
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

COLLECTION_NAME = "SIAPE"

# Inicializa clientes
openai_client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Função para gerar embedding de uma pergunta
def embed_query(query):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    return response.data[0].embedding

# Função para buscar no Qdrant
def search_qdrant(query, top_k=5):
    vector = embed_query(query)
    search_result = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=top_k
    )
    return search_result

# Teste
if __name__ == "__main__":
    pergunta = input("❓ Sua pergunta: ")
    resultados = search_qdrant(pergunta)

    print("\n🔍 Resultados mais relevantes:")
    for i, result in enumerate(resultados):
        print(f"\n{i+1}. Score: {result.score:.4f}")
        print(result.payload["text"])