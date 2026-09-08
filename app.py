import os
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

def dapatkan_api_key():
    """Mengambil API Key dari st.secrets (Streamlit Cloud/Hugging Face) atau .env (Lokal)."""
    api_key = None
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
        
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
    return api_key

api_key_aktif = dapatkan_api_key()

@st.cache_resource
def load_wayang_model():
    model = tf.keras.models.load_model("mobilenetv2_wayang_finetuned.keras")
    model.build(input_shape=(None, 299, 299, 3))
    return model

model = load_wayang_model()

@st.cache_resource
def load_rag_resources(api_key):
    if not api_key:
        return None, None, None
        
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    nama_folder_faiss = "faiss_wayang_index"
    
    vectorstore = FAISS.load_local(
        nama_folder_faiss, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.3, google_api_key=api_key)
    
    system_prompt = (
        "Kamu adalah asisten virtual ahli pewayangan Jawa. "
        "Catatan sinonim ejaan tokoh: Anoman = Hanoman, Bima = Werkudara, Gatotkaca = Gathotkaca/Tetuka, Kresna = Krisna, Durna = Drona. "
        "Gunakan potongan konteks berikut untuk menjawab pertanyaan pengguna. "
        "Jawablah pertanyaan pengguna HANYA berdasarkan potongan konteks yang diberikan di bawah ini. JANGAN gunakan pengetahuan di luar konteks. Jika konteks yang diberikan adalah tentang Nakula dan pengguna bertanya tentang Sadewa, ingatkan pengguna untuk mengubah sub-fokus ke 'Khusus Sadewa'. "
        "Jika kamu tidak tahu jawabannya berdasarkan konteks, katakan saja kamu tidak tahu. "
        "Konteks:\n{context}"
    )
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    return vectorstore, llm, prompt_template

# Validasi ketersediaan API Key sebelum memuat RAG
if not api_key_aktif:
    st.error("⚠️ **GOOGLE_API_KEY Tidak Ditemukan!**")
    st.warning("Mohon pastikan file `.env` di lokal berisi `GOOGLE_API_KEY=...` atau atur di `Secrets` platform cloud hosting Anda.")
    st.stop()

try:
    vectorstore, llm, prompt_template = load_rag_resources(api_key_aktif)
except Exception as e:
    st.error(f"⚠️ **Gagal Memuat Resources RAG**: {e}")
    st.stop()

# -----------------------------------------
# 1. KONFIGURASI DAFTAR KARAKTER (22 KELAS)
# -----------------------------------------
DAFTAR_WAYANG = [
    'abimanyu', 'anoman', 'arjuna', 'bagong', 'baladewa', 'bima', 
    'buta', 'cakil', 'durna', 'dursasana', 'duryudana', 'gareng', 
    'gatotkaca', 'karna', 'kresna', 'nakula_sadewa', 'patih_sabrang', 
    'petruk', 'puntadewa', 'semar', 'sengkuni', 'togog'
]

def dapatkan_list_karakter(slug, sub_opsi="Keduanya"):
    """Fungsi Memetakan slug CV ke list nama karakter untuk filter RAG."""
    if slug == 'nakula_sadewa':
        if sub_opsi == "Khusus Nakula":
            return ['nakula']
        elif sub_opsi == "Khusus Sadewa":
            return ['sadewa']
        else:
            return ['nakula', 'sadewa']
    return [slug]

def buat_sapaan_disclaimer(slug, confidence):
    """Fungsi pembentuk sapaan disclaimer bertingkat berdasarkan confidence score."""
    nama_tampil = "Nakula & Sadewa" if slug == "nakula_sadewa" else slug.replace('_', ' ').title()
    pct = confidence * 100
    
    if confidence > 0.80:
        return (
            f"🟢 **Hasil Identifikasi: {nama_tampil}** (Keyakinan: {pct:.1f}%)\n\n"
            f"Halo! Saya sangat yakin gambar ini adalah **{nama_tampil}**. Ada yang ingin kamu ketahui tentang silsilah atau kisahnya?"
        )
    elif confidence >= 0.50:
        return (
            f"🟡 **Hasil Identifikasi: {nama_tampil}** (Keyakinan: {pct:.1f}%)\n\n"
            f"Halo! Pindaian gambar ini lumayan mirip dengan **{nama_tampil}**, tetapi saya sedikit ragu. "
            f"Jika tokoh ini kurang tepat, kamu dapat mengoreksinya pada menu **2. Konfirmasi Karakter** di bawah. Apa yang ingin kamu tanyakan?"
        )
    else:
        return (
            f"🔴 **Hasil Identifikasi: {nama_tampil}** (Keyakinan Rendah: {pct:.1f}%)\n\n"
            f"Halo! Kualitas atau sudut foto kurang jelas. Tebakan terbaik saya adalah **{nama_tampil}**. "
            f"**Disarankan untuk memeriksa dan memilih tokoh yang benar pada menu 2. Konfirmasi Karakter di bawah** sebelum memulai diskusi."
        )

# -----------------------------------------
# 2. INISIALISASI SESSION STATE
# -----------------------------------------
if 'karakter_prediksi_cv' not in st.session_state:
    st.session_state.karakter_prediksi_cv = None
if 'confidence_cv' not in st.session_state:
    st.session_state.confidence_cv = None
if 'karakter_aktif_slug' not in st.session_state:
    st.session_state.karakter_aktif_slug = DAFTAR_WAYANG[0]
if 'sub_opsi_nakula_sadewa' not in st.session_state:
    st.session_state.sub_opsi_nakula_sadewa = "Keduanya (Nakula & Sadewa)"
if 'karakter_aktif' not in st.session_state:
    st.session_state.karakter_aktif = dapatkan_list_karakter(DAFTAR_WAYANG[0])
if 'pesan_chat' not in st.session_state:
    st.session_state.pesan_chat = [{
        "role": "assistant", 
        "content": "Selamat datang! Silakan unggah foto wayang di atas untuk memindai, atau langsung pilih tokoh pada menu **2. Konfirmasi Karakter** untuk mulai berdiskusi."
    }]

st.title("Wayang AI: Deteksi & Eksplorasi Kisah")

# -----------------------------------------
# 3. ALUR UPLOAD & PREDIKSI CV
# -----------------------------------------
st.header("1. Pindai Gambar Wayang")
uploaded_file = st.file_uploader("Unggah foto wayang di sini...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Foto yang diunggah", use_container_width=True)
    
    if st.button("Mulai Identifikasi"):
        img = image.resize((299, 299))
        img_array = np.array(img)
        
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,)*3, axis=-1)
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        
        predictions = model.predict(img_array, verbose=0)[0]
        max_index = np.argmax(predictions)
        
        prediksi_slug = DAFTAR_WAYANG[max_index]
        confidence = float(predictions[max_index])
        
        st.session_state.karakter_prediksi_cv = prediksi_slug
        st.session_state.confidence_cv = confidence
        st.session_state.karakter_aktif_slug = prediksi_slug
        st.session_state.sub_opsi_nakula_sadewa = "Keduanya (Nakula & Sadewa)"
        st.session_state.karakter_aktif = dapatkan_list_karakter(prediksi_slug)
        
        sapaan = buat_sapaan_disclaimer(prediksi_slug, confidence)
        
        st.session_state.pesan_chat = [{"role": "assistant", "content": sapaan}]
        st.rerun()

st.divider()

# -----------------------------------------
# 4. ALUR PILIHAN MANUAL & SINKRONISASI KOREKSI
# -----------------------------------------
st.header("2. Konfirmasi Karakter")

try:
    default_idx = DAFTAR_WAYANG.index(st.session_state.karakter_aktif_slug)
except ValueError:
    default_idx = 0

pilihan_manual = st.selectbox(
    "Pilih karakter utama:",
    options=DAFTAR_WAYANG,
    index=default_idx,
    format_func=lambda x: "Nakula & Sadewa (Sepasang)" if x == "nakula_sadewa" else x.replace('_', ' ').title()
)

if pilihan_manual != st.session_state.karakter_aktif_slug:
    st.session_state.karakter_aktif_slug = pilihan_manual
    st.session_state.sub_opsi_nakula_sadewa = "Keduanya (Nakula & Sadewa)"
    st.session_state.karakter_aktif = dapatkan_list_karakter(pilihan_manual)
    
    nama_koreksi = "Nakula & Sadewa" if pilihan_manual == "nakula_sadewa" else pilihan_manual.replace('_', ' ').title()
    info_ganti = f"🔄 *Karakter dikoreksi secara manual menjadi **{nama_koreksi}**. Pembahasan chatbot sekarang difokuskan pada {nama_koreksi}.*"
    st.session_state.pesan_chat.append({"role": "assistant", "content": info_ganti})
    st.rerun()

if st.session_state.karakter_aktif_slug == 'nakula_sadewa':
    st.markdown("**Sub-Fokus Diskusi Nakula / Sadewa:**")
    sub_opsi = st.radio(
        "Pilih spesifikasi pembahasan chatbot:",
        options=["Keduanya (Nakula & Sadewa)", "Khusus Nakula", "Khusus Sadewa"],
        index=["Keduanya (Nakula & Sadewa)", "Khusus Nakula", "Khusus Sadewa"].index(st.session_state.sub_opsi_nakula_sadewa),
        horizontal=True
    )
    
    if sub_opsi != st.session_state.sub_opsi_nakula_sadewa:
        st.session_state.sub_opsi_nakula_sadewa = sub_opsi
        st.session_state.karakter_aktif = dapatkan_list_karakter('nakula_sadewa', sub_opsi)
        st.session_state.pesan_chat.append({
            "role": "assistant", 
            "content": f"🎯 *Sub-fokus diskusi dispesifikkan ke: **{sub_opsi}**.*"
        })
        st.rerun()

st.divider()

# -----------------------------------------
# 5. ALUR CHATBOT RAG
# -----------------------------------------
st.header("3. Ruang Diskusi")

for msg in st.session_state.pesan_chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.karakter_aktif_slug == "nakula_sadewa":
    prompt_label = st.session_state.sub_opsi_nakula_sadewa
else:
    prompt_label = st.session_state.karakter_aktif_slug.replace('_', ' ').title()

prompt = st.chat_input(f"Tanyakan sesuatu tentang {prompt_label}...")

if prompt:
    st.session_state.pesan_chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Berpikir..."):
            target_karakter = st.session_state.karakter_aktif
            top_k = 8 if len(target_karakter) > 1 else 5
            
            filter_fn = lambda metadata: metadata.get("Karakter_CV") in target_karakter
            
            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": top_k,
                    "filter": filter_fn
                }
            )
            
            question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            query_input = prompt
            if "anoman" in prompt.lower() and "hanoman" not in prompt.lower():
                query_input = prompt + " (Hanoman)"
            
            try:
                response = rag_chain.invoke({"input": query_input})
                jawaban_gemini = response['answer']
                st.markdown(jawaban_gemini)
                st.session_state.pesan_chat.append({"role": "assistant", "content": jawaban_gemini})
            except Exception as e:
                pesan_error = f"⚠️ **Terjadi kendala saat menghubungi API Gemini**: {e}"
                st.error(pesan_error)