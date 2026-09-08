# Rangkuman Proyek: Sistem Pengenalan Karakter Wayang (CV) + Chatbot RAG (v4 Final)

## Konteks Umum
- Mahasiswa Informatika ITB Asia Malang, tugas akhir (TA) jalur jurnal (target SINTA 4), sekaligus sedang menyusun proposal untuk conference internal kampus dan bersiap menghadapi ujian judul.
- Ide dasar proyek dari dosen pembimbing; mahasiswa mengeksekusi dan mengembangkan lewat banyak keputusan teknis independen sepanjang proses.

## STACK TEKNIS FINAL

### Computer Vision
- Arsitektur: **MobileNetV2** (transfer learning dari ImageNet)
- Resolusi input: **299x299 px**
- Fine-tuning: **unfreeze 30 layer terakhir**
- Hasil akhir: **91% accuracy, macro F1-score 0.912** (evaluasi Colab)
- Format model: `.keras` untuk deployment Python (`mobilenetv2_wayang_finetuned.keras`)
- Jumlah kelas terlatih: **22 kelas** (lihat bagian "Keputusan 19 vs 22 kelas & Posisi Tokoh Durna")
- Preprocessing di model: augmentasi (layer index 0, aman dibuang saat inference) + Rescaling (layer index 1, WAJIB tetap ada — jangan pernah dibuang saat load model untuk inference)

### Chatbot — Klasifikasi Metode (PENTING untuk proposal/sidang)
- **Metode: Retrieval-Augmented Generation (RAG)** — BUKAN rule-based/intent classification, dan BUKAN fine-tuning/transfer learning model generatif seperti T5.
- Prinsip kerja: model generatif (Gemini) **tidak diubah bobotnya** ("frozen LLM") — pengetahuan domain wayang disuntikkan lewat konteks hasil retrieval di setiap prompt ("in-context learning"), bukan ditanam permanen lewat training ulang.
- Embedding: **`gemini-embedding-001`** (versi stabil, DIKONFIRMASI final)
- Orkestrasi: **LangChain** (`langchain_classic` untuk `create_retrieval_chain`, `create_stuff_documents_chain` — API v1.0+)
- Vector store: **FAISS** (dengan caching lokal `faiss_wayang_index`)
- LLM Generation: **Gemini API** (`gemini-3.1-flash-lite`, temperature=0.3, free tier cukup untuk seluruh development & demo)
- Knowledge base: CSV (`knowledge_base_wayang_new.csv`), dikelola manual via Excel (header section pakai `##` bukan `===`)

### Deployment
- **Platform: Python + Streamlit** (`app.py`)
- Target hosting: **Streamlit Community Cloud** (seluruh dokumen telah diselaraskan ke Streamlit Cloud)
- Penanganan API Key: Otomatis mendeteksi `st.secrets["GOOGLE_API_KEY"]` untuk Cloud dan fallback ke `.env` untuk pengujian Lokal.
- Keamanan: File `.env` masuk ke `.gitignore` agar kredensial API tidak bocor.

## PROGRES COMPUTER VISION

### Dataset
- 6.576 citra, 22 kelas, dari Kaggle (foto wayang kulit fisik asli)
- Distribusi: 192–400 gambar/kelas (imbalance ringan, ~2:1)
- Split: stratified 70:15:15 (train:val:test), seed=42, pakai `splitfolders`

### Bug yang sudah diperbaiki
- Folder output split sempat bersarang di dalam folder dataset sumber, menyebabkan folder split terbaca sebagai "kelas ke-23" (`hasil_dataset_split`) di confusion matrix. Sudah diperbaiki dengan memisahkan `OUTPUT_DIR`.

### Eksperimen & hasil (evaluasi Colab, kondisi terkontrol)
| Eksperimen | Accuracy | Macro F1 |
|---|---|---|
| Baseline 224x224, frozen | 77% | 0.77 |
| 299x299, frozen | 85% | 0.84 (patih_sabrang justru memburuk, F1 0.59) |
| **299x299 + unfreeze 30 layer (FINAL)** | **91%** | **0.912** |

- Confusion matrix mengonfirmasi cluster "karakter gagah/Kurawa" (dursasana, duryudana, karna, gatotkaca, patih_sabrang, sengkuni) sebagai kelompok inter-class similarity tinggi — terbukti BUKAN karena resolusi rendah, melainkan genuine kemiripan visual siluet.
- **Catatan Penting Tokoh Durna**: Durna mencapai F1 sempurna (1.0) — kelas dengan performa terbaik di antara 22 kelas. Eliminasi Durna dari pembahasan 19 karakter murni pertimbangan signifikansi naratif cerita wayang oleh dosen, BUKAN karena kurangnya kemampuan model.

### Kelas khusus `nakula_sadewa`
- Dataset training untuk kelas ini SELALU berisi foto SEPASANG wayang (desain dataset yang disengaja, bukan bug). Model tidak bisa mendeteksi Nakula/Sadewa individual. Ditangani di layer aplikasi (`app.py`) lewat pemetaan list metadata RAG dan UI Sub-fokus.

### Referensi pembanding yang ditemukan mahasiswa: EfficientNetB2
- Ditemukan paper "Penerapan Arsitektur CNN-EfficientNetB2 dengan Transfer Learning pada Klasifikasi Gambar Tokoh Wayang Kulit" — melaporkan akurasi **96,6%**, lebih tinggi dari MobileNetV2 (91%) pada penelitian ini.
- Strategi framing yang disepakati: BUKAN sebagai kekalahan, tapi sebagai **trade-off akurasi vs efisiensi komputasi** — MobileNetV2 dipilih karena parameter lebih ringan (5.395.332 params) dan lebih cocok untuk deployment real-time dengan sistem RAG yang juga butuh resource.

### Pengujian lapangan (field testing, di luar Colab)
- Setup: kamera HP + tripod, jarak tetap 25 cm dari citra wayang hasil CETAK (print, bukan wayang fisik asli — karena tidak ada akses; bukan foto dari layar — karena artefak moiré/glare).
- Hasil: **38 gambar diuji (114 data point dari 3x scan/gambar), 26/38 benar = ~68,4% akurasi** — turun signifikan dari 91% Colab.
- **Temuan 1 — cluster gagah/Kurawa terkonfirmasi ulang, lebih parah**: karna→kresna, duryudana→patih_sabrang/arjuna, dursasana→buta/sengkuni, sengkuni→petruk.
- **Temuan 2 — cluster BARU "ksatria halus"**: Arjuna, Puntadewa, Abimanyu, Nakula-Sadewa saling tertukar.
- **Temuan 3 — paradoks confidence**: prediksi BENAR dengan confidence sangat rendah (durna 31-37%, petruk 38-42%, togog 39-42%) vs prediksi SALAH dengan confidence sedang (karna→kresna 49%). Solusi: Disambiguasi via disclaimer bertingkat & koreksi manual.
- **Temuan 4 — sensitivitas jarak/framing kamera**: Karna sempat salah terdeteksi Arjuna saat difoto jauh, benar saat didekatkan. Solusi: standarisasi jarak 25cm + tripod.

## PROGRES CHATBOT RAG & SISTEM INTEGRASI

### Temuan cross-class retrieval & Solusi Metadata Filter (SELESAI)
- **Masalah**: pure semantic search kadang mengambil chunk dari kelas lain yang mirip secara tekstual.
- **Solusi Terimplementasi di `app.py`**: Metadata filter kustom berbasis `lambda metadata: metadata.get("Karakter_CV") in target_karakter`.
- **Top-K Dinamis & Penanganan Varian Ejaan**:
  - Karakter Tunggal: `top_k = 5` (memastikan chunk sekunder seperti 'Anggota Keluarga' tidak terpotong akibat variasi ejaan seperti *Anoman* vs *Hanoman*).
  - Kasus `nakula_sadewa`: `top_k = 8` dengan pemetaan list `['nakula', 'sadewa']`.

### Kurasi knowledge base — kasus khusus
- **Wikipedia extraction**: strategi adaptif (≤6000/7500 karakter = ambil utuh, lebih = ringkasan + section terfilter). Header section pakai `## Nama Section ##`.
- **Patih Sabrang**: tokoh srambahan/arketipe, nama tidak baku. Dikonfirmasi via Ki Wahyu Dunung (dalang bersertifikat). Draft manual sudah masuk CSV.
- **Buta**: golongan raksasa (bukan individu tunggal), mencakup Kumbakarna, Buto Raton, Buto Punuk, Brojodento.
- **Nakula & Sadewa**: dipisah jadi 2 baris knowledge base terpisah (`nakula` dan `sadewa`) meski CV 1 kelas gabungan.

### Arsitektur Merger CV + RAG & UI Streamlit (SELESAI)
- `session_state.karakter_prediksi_cv` — **Klarifikasi Logging**: Menyimpan hasil prediksi mentah model CV & confidence score murni untuk analisis/audit logging sesi (melihat frekuensi koreksi manual user), dan nilainya **tidak pernah di-overwrite** meskipun pengguna melakukan koreksi manual.
- `session_state.karakter_aktif` — LIST nama karakter yang dikirim ke filter FAISS (diperbarui saat scan atau koreksi manual).
- **Limitasi Desain Perpindahan Konteks Karakter (PENTING untuk Bab IV)**: Sistem sengaja tidak mendukung perpindahan konteks karakter di tengah sesi percakapan hanya lewat ketikan teks bebas. Pengguna HARUS memilih secara eksplisit via menu *2. Konfirmasi Karakter* (dropdown / radio button) atau melakukan scan ulang. Ini adalah *explicit state control* untuk mencegah kebocoran RAG.
- **Fitur Sub-Fokus Nakula/Sadewa**: Menyediakan radio button UI (*Keduanya*, *Khusus Nakula*, *Khusus Sadewa*).
- **Disclaimer Bertingkat Confidence CV**:
  - Tinggi (>80%): Badge 🟢 tanpa keraguan.
  - Sedang (50-80%): Badge 🟡 dengan disclaimer halus untuk memeriksa ulang konfirmasi.
  - Rendah (<50%): Badge 🔴 dengan disclaimer tegas meminta pemeriksaan menu konfirmasi.
- **System Prompt Strict Grounding (Opsi B)**:
  Instruksi ketat bahwa Gemini HANYA boleh menjawab berdasarkan potongan konteks ter-retrieve, menolak berhalusinasi dari ingatan internet umum, serta dilengkapi catatan sinonim ejaan wayang Jawa (*Anoman = Hanoman, Bima = Werkudara, Gatotkaca = Gathotkaca/Tetuka, Kresna = Krisna, Durna = Drona*).

## STATUS INTEGRASI TEKNIS (100% SELESAI & DEPLOYMENT-READY)
- Pipeline RAG standalone: **SUDAH SELESAI**
- Model CV standalone: **SUDAH SELESAI**
- Metadata filter (solusi cross-class): **SUDAH DIIMPLEMENTASIKAN & DIUJI (100% WORKING)**
- Integrasi CV + RAG dalam 1 aplikasi Streamlit (`app.py`): **SUDAH SELESAI & BERJALAN TERUJI LOKAL**
- Penanganan API Key Dual-Environment (`st.secrets` & `.env`) & Graceful Error Handling (`st.error`): **SUDAH SELESAI**

## PROGRES PROPOSAL TA / CONFERENCE INTERNAL KAMPUS

### Format
Template proposal internal ITB Asia Malang (7 bagian: Tema, Judul, Latar Belakang, Rumusan Masalah, Tujuan Penelitian, Metode Penelitian, Daftar Rujukan).

### Draft Proposal
- **Tema**: Integrasi Sistem Cerdas (Computer Vision dan Retrieval-Augmented Generation)
- **Judul**: "Pengenalan Karakter Wayang Jawa Menggunakan Arsitektur MobileNetV2 dan Generasi Teks Berbasis RAG"
- **Latar Belakang**: 3 paragraf (Fenomena penurunan minat & kesulitan identifikasi visual; Solusi CV MobileNetV2; Solusi RAG Gemini).
- **Rumusan Masalah & Tujuan Penelitian** (3 poin berkorespondensi 1:1):
  1. Implementasi & evaluasi MobileNetV2 transfer learning 22 kelas dengan fokus pembahasan 19 karakter utama.
  2. Mekanisme integrasi filter metadata CV➔FAISS (termasuk penanganan kelas khusus Nakula-Sadewa).
  3. Efektivitas RAG berbasis Gemini dengan Strict Grounding dalam menekan halusinasi.

### Keputusan penting: 19 vs 22 kelas & Posisi Tokoh Durna
- Dosen pembimbing meminta fokus pembahasan laporan ke 19 karakter utama, mengeliminasi **Patih Sabrang, Durna, Cakil** demi signifikansi naratif cerita wayang.
- **Durna F1 1.0 (Sempurna)**: Eliminasi Durna murni pertimbangan signifikansi naratif tokoh dalam lakon wayang, BUKAN karena performa model.
- **Model tetap 22 kelas terlatih** (TIDAK retraining ulang). Pembatasan 19 hanya pada cakupan pembahasan laporan.
- Narasi kronologis: Training 22 kelas dilakukan sebagai eksplorasi awal sebelum konsultasi; hasil confusion matrix 22 kelas menjadi bahan diskusi yang mengarahkan pada rekomendasi penyempitan fokus pembahasan.

## PERSIAPAN UJIAN JUDUL

Dosen memberi bocoran 4 poin wajib bisa dijelaskan saat presentasi:
1. **Masalah**: Pelajar/masyarakat kesulitan identifikasi visual wayang (ornamen rumit, kemiripan siluet) ➔ menurunnya minat generasi muda. Chatbot LLM murni berisiko halusinasi untuk budaya lokal spesifik.
2. **Tujuan**: (1) Model klasifikasi citra akurat, (2) Chatbot faktual berbasis RAG, (3) Integrasi keduanya jadi sistem utuh.
3. **Solusi/metode**: Sistem 2 tahap — CV MobileNetV2 mendeteksi karakter ➔ konteks otomatis untuk Chatbot RAG berbasis Gemini.
4. **Dataset**: 6.576 citra, 22 kelas, Kaggle, split 70:15:15 stratified. Knowledge base: ekstraksi adaptif Wikipedia + kurasi manual.

## NEXT STEPS PRIORITAS REALISTIS
1. **Deploy ke Streamlit Community Cloud**:
   Upload/push repositori ke GitHub dan hubungkan ke Streamlit Community Cloud (konfigurasikan `GOOGLE_API_KEY` pada menu Secrets).
2. **Selesaikan Dokumen Proposal (Word)**:
   - Kompilasi dan lengkapi Daftar Rujukan final.
   - Verifikasi sitasi mandiri.
   - Finalisasi keputusan penyesuaian target audiens ke "pelajar".
   - Buat 2 flowchart editable (CV & Integrasi RAG) sesuai format Word.
3. **Latihan Presentasi Ujian Judul**:
   - Kuasai 4 poin bocoran dosen.
   - Antisipasi pertanyaan susulan (kenapa 19/22 kelas, penurunan akurasi lapangan 68.4%, dan penanganan confidence rendah).
