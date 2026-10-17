from qdrant_client import QdrantClient
import ollama

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "sangam_docs"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1:8b"

client = QdrantClient(url=QDRANT_URL)

def search(query, top_k=3):
    q_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_embedding,
        limit=top_k
    ).points
    
    print(f"\n🔍 Query: {query}")
    print(f"Found {len(results)} relevant chunks:\n")
    
    contexts = []
    for i, hit in enumerate(results):
        text = hit.payload["text"]
        score = hit.score
        contexts.append(text)
        print(f"--- Chunk {i+1} (Score: {score:.4f}) ---")
        print(f"{text[:400]}...\n")
    
    return contexts

def ask_llm(query, contexts):
    context_text = "\n\n".join(contexts)
    prompt = f"""You are Sangam-Brain. Answer ONLY from the context below.

Context:
{context_text}

Question: {query}

Rules:
- If answer is not in context, say "Not found in my September 2026 PDF"
- Answer in simple bullet points"""

    print("🤖 Llama is thinking...\n")
    response = ollama.chat(model=LLM_MODEL, messages=[{"role":"user","content":prompt}])
    print("✅ ANSWER:\n")
    print(response['message']['content'])
    return response['message']['content']

if __name__ == "__main__":
    my_question = "What is SAR technology mentioned in current affairs?"
    ctx = search(my_question, top_k=3)
    ask_llm(my_question, ctx)