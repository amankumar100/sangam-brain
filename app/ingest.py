import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import ollama
from pypdf import PdfReader
import uuid

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "sangam_docs"
EMBED_MODEL = "nomic-embed-text"

client = QdrantClient(url=QDRANT_URL)

def create_collection():
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        print(f"✅ Collection '{COLLECTION_NAME}' created")
    else:
        print(f"ℹ️ Collection '{COLLECTION_NAME}' already exists")

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def ingest_pdf(pdf_path):
    print(f"📄 Reading {pdf_path}...")
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    
    print(f"📝 Total chars: {len(full_text)}")
    chunks = chunk_text(full_text)
    print(f"✂️ Split into {len(chunks)} chunks")

    points = []
    for i, chunk in enumerate(chunks):
        # Create embedding via Ollama
        response = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)
        embedding = response["embedding"]
        
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": chunk, "source": pdf_path, "chunk_id": i}
        ))
        print(f"  Embedded chunk {i+1}/{len(chunks)}")

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Ingested {len(points)} chunks to Qdrant!")

if __name__ == "__main__":
    create_collection()
    # Put any PDF in data/ folder and update name below
    pdf_file = "data/sample.pdf"
    if os.path.exists(pdf_file):
        ingest_pdf(pdf_file)
    else:
        print(f"⚠️ Put a PDF at {pdf_file} to test. Collection is ready anyway.")