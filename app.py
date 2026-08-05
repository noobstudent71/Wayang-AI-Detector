import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

@st.cache_resource
def load_wayang_model():
    model = tf.keras.models.load_model("mobilenetv2_wayang_finetuned.keras")
    model.build(input_shape=(None, 299, 299, 3))
    return model
model = load_wayang_model()

@st.cache_resource
def load_rag_resources():
    # 1. Setup Embedding & Vectorstore
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    nama_folder_faiss = "faiss_wayang_index"
    
    # Memuat index FAISS lokal
    vectorstore = FAISS.load_local(
        nama_folder_faiss, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # 2. Setup LLM Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.3)
    
    # 3. Setup Prompt Template
    system_prompt = (
        "Kamu adalah asisten virtual ahli pewayangan Jawa. "
        "Gunakan potongan konteks berikut untuk menjawab pertanyaan pengguna. "
        "Jika kamu tidak tahu jawabannya berdasarkan konteks, katakan saja kamu tidak tahu. "
        "Konteks:\n{context}"
    )
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    return vectorstore, llm, prompt_template
# Panggil fungsi inisialisasi
vectorstore, llm, prompt_template = load_rag_resources()

# Import library TensorFlow, LangChain, dan Gemini kamu di sini

# -----------------------------------------
# 1. KONFIGURASI DAFTAR KARAKTER
# -----------------------------------------
# Sesuaikan dengan 22 label yang kamu miliki, ingat 'nakula_sadewa' digabung
DAFTAR_WAYANG = [
    'abimanyu', 'anoman', 'arjuna', 'bagong', 'baladewa', 'bima', 
    'buta', 'cakil', 'durna', 'dursasana', 'duryudana', 'gareng', 
    'gatotkaca', 'karna', 'kresna', 'nakula_sadewa', 'patih_sabrang', 
    'petruk', 'puntadewa', 'semar', 'sengkuni', 'togog'
    # Tambahkan sisanya sampai 22 karakter
]

# -----------------------------------------
# 2. INISIALISASI SESSION STATE
# -----------------------------------------
# 'brankas memori' agar data tidak hilang saat layar refresh
if 'karakter_prediksi_cv' not in st.session_state:
    st.session_state.karakter_prediksi_cv = None
if 'karakter_aktif' not in st.session_state:
    st.session_state.karakter_aktif = DAFTAR_WAYANG[0] # Default ke item pertama
if 'pesan_chat' not in st.session_state:
    st.session_state.pesan_chat = []

st.title("Wayang AI: Deteksi & Eksplorasi Kisah")

# -----------------------------------------
# 3. ALUR UPLOAD & PREDIKSI CV
# -----------------------------------------
st.header("1. Pindai Gambar Wayang")
uploaded_file = st.file_uploader("Unggah foto wayang di sini...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file)
    st.image(image, caption="Foto yang diunggah", use_container_width=True)
    
    if st.button("Mulai Identifikasi"):
          # --- LOGIKA MODEL TENSORFLOW/MOBILENETV2 ---
        # 1. Preprocessing gambar dari PIL Image ke array NumPy & ubah ukuran ke 299x299
        img = image.resize((299, 299))
        img_array = np.array(img)
        
        # Pastikan gambar memiliki format RGB dan konversi ke batch size (1, 299, 299, 3)
        if len(img_array.shape) == 2:  # Jika gambar grayscale
            img_array = np.stack((img_array,)*3, axis=-1)
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        
        # 2. Lakukan Prediksi
        predictions = model.predict(img_array, verbose=0)[0]
        max_index = np.argmax(predictions)
        
        # 3. Ambil Hasil Kelas Prediksi & Skor Confidence
        prediksi = DAFTAR_WAYANG[max_index]
        confidence = float(predictions[max_index])
        
        # Simpan hasil tebakan murni dari kamera
        st.session_state.karakter_prediksi_cv = prediksi
        # Langsung jadikan tebakan ini sebagai karakter yang aktif dibahas
        st.session_state.karakter_aktif = prediksi
        
        # Logika Tingkat Confidence & Sapaan Pertama (Disclaimer)
        if confidence > 0.80:
            sapaan = f"Halo! Dari hasil pindaian, saya sangat yakin ini adalah **{prediksi.replace('_', ' ').title()}** (Akurasi: {confidence*100:.1f}%). Ada yang ingin kamu ketahui tentang kisahnya?"
        elif confidence >= 0.50:
            sapaan = f"Halo! Gambar ini lumayan mirip dengan **{prediksi.replace('_', ' ').title()}** (Akurasi: {confidence*100:.1f}%), tapi saya sedikit ragu. Jika salah, silakan koreksi di menu bawah ya. Apa yang ingin kamu tanyakan?"
        else:
            sapaan = f"Halo! Pindaian kurang jelas, tebakan terbaik saya adalah **{prediksi.replace('_', ' ').title()}** (Akurasi: {confidence*100:.1f}%). Mohon pastikan lagi tokoh yang tepat di bawah ini. Apa yang bisa saya bantu?"
        
        # Bersihkan riwayat chat lama dan masukkan pesan sapaan baru
        st.session_state.pesan_chat = [{"role": "assistant", "content": sapaan}]
        st.rerun() # Refresh layar agar UI update

st.divider()

# -----------------------------------------
# 4. ALUR PILIHAN MANUAL (DROPDOWN KOREKSI)
# -----------------------------------------
st.header("2. Konfirmasi Karakter")

# Cari indeks dari karakter aktif saat ini untuk ditampilkan di dropdown
try:
    default_idx = DAFTAR_WAYANG.index(st.session_state.karakter_aktif)
except ValueError:
    default_idx = 0

pilihan_manual = st.selectbox(
    "Pilih karakter yang ingin didiskusikan (Bisa untuk koreksi atau jalan pintas):",
    options=DAFTAR_WAYANG,
    index=default_idx
)

# Jika pengguna mengganti pilihan di dropdown, timpa variabel karakter_aktif
if pilihan_manual != st.session_state.karakter_aktif:
    st.session_state.karakter_aktif = pilihan_manual
    # Opsional: Beri tahu pengguna bahwa karakter telah diganti secara manual
    info_ganti = f"*Karakter diubah secara manual menjadi **{pilihan_manual.replace('_', ' ').title()}**.*"
    st.session_state.pesan_chat.append({"role": "assistant", "content": info_ganti})
    st.rerun()

st.divider()

# -----------------------------------------
# 5. ALUR CHATBOT RAG
# -----------------------------------------
st.header("3. Ruang Diskusi")

# Render semua riwayat chat yang ada di session_state
for msg in st.session_state.pesan_chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input teks dari pengguna
prompt = st.chat_input(f"Tanyakan sesuatu tentang {st.session_state.karakter_aktif.replace('_', ' ').title()}...")
if prompt:
    # Tampilkan prompt pengguna di layar
    st.session_state.pesan_chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # --- MASUKKAN LOGIKA RAG LANGCHAIN, FAISS, & GEMINI DI SINI ---
    # PENTING: Gunakan 'st.session_state.karakter_aktif' sebagai filter metadata k=3 kamu
    with st.chat_message("assistant"):
        with st.spinner("Berpikir..."):
            # 1. Setup retriever dengan filter metadata tokoh aktif
            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": 3,
                    "filter": {"Karakter_CV": st.session_state.karakter_aktif}
                }
            )
            
            # 2. Buat RAG pipeline
            question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            # 3. Eksekusi query
            response = rag_chain.invoke({"input": prompt})
            jawaban_gemini = response['answer']
            
            st.markdown(jawaban_gemini)
    
    # Simpan jawaban ke session_state
    st.session_state.pesan_chat.append({"role": "assistant", "content": jawaban_gemini})