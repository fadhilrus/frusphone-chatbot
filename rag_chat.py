import os
import time
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import google.generativeai as genai
from qdrant_client import QdrantClient

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


genai.configure(api_key=GEMINI_API_KEY)

llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=120,
    verify=False
)

def get_embedding(text: str):
    print(f"[EMBEDDING] Generating query embedding...")
    try:
        result = genai.embed_content(
            model="gemini-embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        if result is None:
            raise ValueError("Embedding API returned None")
        if isinstance(result, dict):
            emb = result.get("embedding")
        else:
            emb = getattr(result, "embedding", None)
        if emb is None:
            raise ValueError("No embedding in response")
        print(f"[EMBEDDING] Query embedding generated (dim={len(emb)})")
        return emb
    except Exception as e:
        print(f"Embedding error: {e}")
        raise

def chat_llm(prompt: str):
    response = llm_client.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def search_products(query: str, limit: int = 3):
    query_embedding = get_embedding(query)
    for attempt in range(3):
        try:
            search_result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding,
                limit=limit
            )
            return search_result.points
        except Exception as e:
            print(f"Qdrant query attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise

def format_context(results):
    context = "Product Catalog:\n\n"
    for i, result in enumerate(results, 1):
        p = result.payload
        context += f"{i}. {p['name']}\n"
        context += f"   Description: {p['description']}\n"
        context += f"   Price: Rp {int(p['price']):,}\n"
        context += f"   Stok: {'tersedia' if p['available'] > 0 else 'tidak tersedia'} ({p['available']} unit)\n\n"
    return context

def chat(query: str):
    print(f"Searching for: {query}")
    results = search_products(query)
    
    if not results:
        return "Maaf, tidak ada produk yang ditemukan."
    
    context = format_context(results)
    
    prompt = f"""Anda adalah asisten penjualan iPhone yang helpful. Gunakan informasi produk di bawah ini untuk menjawab pertanyaan customer. Untuk informasi pembayaran anda dapat menjawab pada website frusphone.

{context}

Pertanyaan customer: {query}
Jawab dalam bahasa Indonesia yang natural dan informatif. Jangan langsung menyebut harga, tapi sebutkan keunggulan produk tersebut.
ATURAN PENTING: Untuk status stok, WAJIB hanya menggunakan kata 'tersedia' jika stok > 0, dan 'tidak tersedia' jika stok = 0. DILARANG menggunakan kata lain seperti 'siap kirim', 'ready stock', 'available', atau apapun di luar dua kata itu."""
    
    return chat_llm(prompt)

if __name__ == "__main__":
    print("RAG Chatbot ready! Ketik pertanyaan atau 'exit' untuk keluar.\n")
    
    while True:
        query = input("Anda: ")
        if query.lower() == "exit":
            break
        
        answer = chat(query)
        print(f"\nBot: {answer}\n")