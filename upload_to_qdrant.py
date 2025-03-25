import os
import json
import uuid
import time
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from openai import OpenAI
import tiktoken

# -----------------------------
# 1. Carrega variáveis do .env
# -----------------------------
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -----------------------------
# 2. Inicializa clientes
# -----------------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

COLLECTION_NAME = "SIAPE"

# Cria coleção se ainda não existir
if not qdrant.collection_exists(collection_name=COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

# -----------------------------
# 3. Função para dividir texto em chunks
# -----------------------------
def chunk_text(text, max_tokens=300, overlap=50):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk)
        chunks.append(chunk_text)
        i += max_tokens - overlap
    return chunks

# -----------------------------
# 4. Lê arquivos JSON e extrai os chunks
# -----------------------------
def process_json_folder(folder_path):
    all_chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                text = json.dumps(data, ensure_ascii=False)  # ou customize aqui
                chunks = chunk_text(text)
                all_chunks.extend(chunks)
    return all_chunks

# -----------------------------
# 5. Gera embeddings com OpenAI
# -----------------------------
def embed_texts(texts):
    embeddings = []
    for i in range(0, len(texts), 10):  # batch de 10
        batch = texts[i:i+10]
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        for j, r in enumerate(response.data):
            embeddings.append((str(uuid.uuid4()), batch[j], r.embedding))
    return embeddings

# -----------------------------
# 6. Executa tudo
# -----------------------------
def main():
    print("📁 Lendo arquivos JSON e preparando chunks...")
    texts = process_json_folder("jsons")
    print(f"🔹 Total de chunks: {len(texts)}")

    print("🧠 Gerando embeddings com OpenAI...")
    embedded = embed_texts(texts)

    print("📤 Enviando para o Qdrant em lotes...")
    batch_size = 50
    for i in range(0, len(embedded), batch_size):
        batch = embedded[i:i+batch_size]
        points = [
            PointStruct(id=uid, vector=embedding, payload={"text": text})
            for uid, text, embedding in batch
        ]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"✅ Enviado batch {i}–{i+len(batch)}")
        time.sleep(0.5)

    print(f"\n🎉 Finalizado: {len(embedded)} vetores enviados para a coleção '{COLLECTION_NAME}'.")

# -----------------------------
# 7. Roda o script
# -----------------------------
if __name__ == "__main__":
    main()