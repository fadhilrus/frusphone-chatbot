import os
import time
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import google.generativeai as genai
from qdrant_client import QdrantClient

# ==========================================
# 1. INITIALIZATION & SETUP
# ==========================================
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
SIMILARITY_THRESHOLD = 0.7

# SESUAIKAN DENGAN URL WEBSITE UTAMA PHP KAMU
PHP_WEBSITE_URL = "http://localhost/frusphone" 

genai.configure(api_key=GEMINI_API_KEY)

llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY
)

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=120,
    verify=False
)

st.set_page_config(
    page_title="Frusphone AI",
    page_icon="📱",
    layout="centered",  # Mengubah layout ke centered agar pas di tengah seperti Gemini
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CUSTOM GEMINI CSS STYLING
# ==========================================
st.markdown("""
<style>
    /* Gradient Background ala Gemini */
    .stApp {
        background-color: #0d0f12;
        background-image: 
            radial-gradient(circle at 50% 100%, #172554 0%, transparent 60%),
            radial-gradient(circle at 50% 0%, #030712 0%, transparent 100%);
        color: #f3f4f6;
    }

    /* Sembunyikan Header bawaan Streamlit agar bersih */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* Styling Teks Hero / Welcome Screen */
    .hero-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-top: 5vh;
        margin-bottom: 5vh;
        text-align: center;
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 600;
        background: linear-gradient(135deg, #60a5fa 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        color: #9ca3af;
        font-size: 1.1rem;
    }

    /* Custom Input Bar ala Gemini */
    div[data-testid="stChatInput"] {
        border-radius: 28px !important;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }

    div[data-testid="stChatInput"] > div {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 28px !important;
    }

    div[data-testid="stChatInput"] textarea,
    div[data-testid="stChatInput"] input,
    div[data-testid="stChatInput"] [contenteditable="true"],
    div[data-testid="stChatInput"] div[data-testid="stChatInputTextArea"] {
        background-color: #1e293b !important;
        color: #f3f4f6 !important;
        caret-color: #f3f4f6 !important;
    }

    /* Custom Message Box */
    .stChatMessage {
        background-color: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        margin-bottom: 1rem;
        text-align: justify;
    }

    /* Desktop: wider sidebar */
    @media (min-width: 769px) {
        section[data-testid="stSidebar"] {
            width: 21rem !important;
            min-width: 21rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. HELPER FUNCTIONS & RAG LOGIC
# ==========================================
def get_embedding(text: str):
    print(f"[1] Generating embedding for query")
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
        print(f"[2] Embedding generated (dim: {len(emb)})")
        return emb
    except Exception as e:
        print(f"Embedding error: {e}")
        raise

def chat_llm(prompt: str):
    print(f"[5] Calling OpenRouter LLM...")
    response = llm_client.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"[6] LLM response received")
    return response.choices[0].message.content

def search_products(query: str, limit: int = 3):
    query_embedding = get_embedding(query)
    print(f"[3] Searching Qdrant for top {limit} results...")
    
    for attempt in range(3):
        try:
            search_result = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding,
                limit=limit,
                score_threshold=SIMILARITY_THRESHOLD
            )
            filtered = [p for p in search_result.points if p.score >= SIMILARITY_THRESHOLD]
            print(f"[4] Found {len(search_result.points)} products from Qdrant (filtered >= {SIMILARITY_THRESHOLD}):")
            for i, point in enumerate(filtered, 1):
                print(f"    {i}. {point.payload['name']} (score: {point.score:.4f})")
            return filtered
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

def process_message(query: str):
    print(f"=== RAG Workflow Started ===")
    print(f"[STEP 1] Processing query: {query}")
    
    if "bayar" in query.lower():
        return "pembayaran dilakukan dengan cash atau transfer, lebih lengkapnya bisa ditanyakan langsung pada petugas toko ya!"
    
    if "pesan" in query.lower() or "order" in query.lower():
        return "pembelian bisa dilakukan melalui website atau langsung ditoko ya! informasi lengkapnya bisa ditanyakan langsung pada petugas toko. Terimakasih"
    
    if "garansi" in query.lower() or "kerusakan" in query.lower():
        return "garansi hanya seminggu setelah pembelian, segala kerusakan yang dilakukan oleh pembeli tidak termasuk garansi. Terima kasih!"
    
    if "frusphone" in query.lower() or "toko" in query.lower():
        return "Frusphone menjual iphone bekas dengan kualitas baik. Saya sebagai chatbot assistant senang bisa membantu. Terima kasih!"
    
    results = search_products(query)
    
    if not results:
        return ("Maaf, saya tidak menemukan produk yang relevan di katalog kami "
                "Silakan tanyakan tentang model iPhone lain atau hubungi kami langsung, Terima Kasih!.")

    context = format_context(results)
    
    prompt = f"""Anda adalah asisten penjualan iPhone yang helpful. Gunakan informasi produk di bawah ini untuk menjawab pertanyaan customer.

{context}

Pertanyaan customer: {query}
Jawab dalam bahasa Indonesia yang natural dan informatif. Berikan penekanan pada keunggulan produk terlebih dahulu, lalu sertakan harga dan ketersediaan stok.
ATURAN PENTING: Untuk status stok, WAJIB hanya menggunakan kata 'tersedia' jika stok > 0, dan 'tidak tersedia' jika stok = 0. DILARANG menggunakan kata lain seperti 'siap kirim', 'ready stock', 'available', atau apapun di luar dua kata itu."""
    
    return chat_llm(prompt)

# ==========================================
# 4. STREAMLIT UI RENDER
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar tetap disiapkan untuk opsi reset / info
with st.sidebar:
    st.title("📱 Frusphone")
    st.caption("AI Sales Assistant Frusphone membantu Anda mencari iPhone yang sesuai dengan kebutuhan dan budget Anda.")
    st.markdown("---")
    

    
    st.markdown("---")
    st.write("**Teknologi:**")
    st.write("- Qdrant Vector Database")
    st.write("- Gemini 001 Embedding")
    st.write("- OpenRouter LLM")
    st.markdown("---")
    if st.button("🗑️ Reset Percakapan", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------
# HEADER UTAMA (DITAROH DI ATAS CHAT/HERO)
# ------------------------------------------
col_head_left, col_head_right = st.columns([4, 1])

with col_head_left:
    st.subheader("📱 Frusphone AI")

with col_head_right:
    st.link_button("🚪 Exit", PHP_WEBSITE_URL, help="Kembali ke Website Utama Frusphone")

st.markdown("---")

# JIKA CHAT MASIH KOSONG: Tampilkan Hero Screen ala Gemini
if len(st.session_state.messages) == 0:
    st.markdown("""
        <div class="hero-container">
            <h1 class="hero-title">Ada yang bisa dibantu?</h1>
            <p class="hero-subtitle">Cari rekomendasi, cek spesifikasi, atau harga iPhone di Frusphone.</p>
        </div>
    """, unsafe_allow_html=True)
else:
    # JIKA SUDAH ADA CHAT: Tampilkan Riwayat Pesan
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message['content'])

# INPUT BAR (TETAP DI BAGIAN BAWAH)
if prompt := st.chat_input("Tanyakan sesuatu tentang iPhone..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Rerun langsung supaya hero screen hilang dan pesan user tampil di layar
    st.rerun()

# Dijalankan setelah rerun jika pesan terakhir berasal dari User
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Mencari informasi, Harap tunggu..."):
            try:
                user_prompt = st.session_state.messages[-1]["content"]
                response = process_message(user_prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})