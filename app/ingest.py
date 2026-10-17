import os
from pypdf import PdfReader
from qdrant_client import QdrantClient, models
import ollama

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "sangam_docs"
EMBED_MODEL = "nomic-embed-text"
PDF_PATH = "data/sample.pdf"

client = QdrantClient(url=QDRANT_URL)

def ensure_collection():
    # Create collection if not exists
    try:
        client.get_collection(COLLECTION_NAME)
        print(f"ℹ Collection '{COLLECTION_NAME}' already exists")
    except Exception:
        print(f"📦 Creating collection '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
        )

def load_pdf(path):
    if not os.path.exists(path):
        print(f"⚠️ Put a PDF at {path}")
        return ""
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    print(f"📄 Reading {path}...")
    print(f"📝 Total chars: {len(text)}")
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    print(f"✂ Split into {len(chunks)} chunks")
    return chunks

def ingest():
    ensure_collection()
    text = load_pdf(PDF_PATH)
    if not text:
        return
    chunks = chunk_text(text)

    # Optional: Clear old data to avoid duplicates
    client.delete_collection(COLLECTION_NAME)
    ensure_collection()

    for i, chunk in enumerate(chunks):
        print(f"  Embedded chunk {i+1}/{len(chunks)}")
        embedding = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)["embedding"]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[models.PointStruct(
                id=i,
                vector=embedding,
                payload={"text": chunk, "source": PDF_PATH, "chunk_id": i}
            )]
        )
    print(f"✅ Ingested {len(chunks)} chunks to Qdrant!")

if __name__ == "__main__":
    ingest()