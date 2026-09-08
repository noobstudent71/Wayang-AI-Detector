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
- Jumlah kelas terlatih: **22 kelas** (lihat bagian "Keputusan 19 vs 22 kelas")
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
- Target hosting: **Streamlit Community Cloud** (BUKAN Hugging Face Spaces — sudah dikonfirmasi final)
- Penanganan API Key: Otomatis mendeteksi `st.secrets["GOOGLE_API_KEY"]` untuk Cloud dan fallback ke `.env` untuk pengujian lokal
- Keamanan: File `.env` masuk ke `.gitignore` agar kredensial API tidak bocor

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
- **Durna mencapai F1 sempurna (1.0) — kelas dengan performa terbaik di seluruh 22 kelas.** Meski demikian, Durna dieliminasi dari fokus pembahasan (bukan dari sistem) atas arahan dosen — murni pertimbangan signifikansi naratif tokoh dalam cerita wayang, BUKAN soal kemampuan model. Jangan pernah jawab "performanya kurang" jika ditanya soal eliminasi Durna — itu kontradiksi dengan data sendiri.

### Kelas khusus `nakula_sadewa`
- Dataset training untuk kelas ini SELALU berisi foto SEPASANG wayang (desain dataset yang disengaja, bukan bug). Model tidak bisa mendeteksi Nakula/Sadewa individual. Ditangani di layer aplikasi (`app.py`) lewat pemetaan list metadata RAG dan UI sub-fokus radio button.

### Referensi pembanding yang ditemukan mahasiswa: EfficientNetB2
- Ditemukan paper "Penerapan Arsitektur CNN-EfficientNetB2 dengan Transfer Learning pada Klasifikasi Gambar Tokoh Wayang Kulit" — melaporkan akurasi **96,6%**, lebih tinggi dari MobileNetV2 (91%).
- Strategi framing yang disepakati: BUKAN sebagai kekalahan, tapi sebagai **trade-off akurasi vs efisiensi komputasi** — MobileNetV2 dipilih karena parameter lebih ringan (5.395.332 params) dan lebih cocok untuk deployment real-time dengan sistem RAG yang juga butuh resource.
- PENTING: paper ini ditemukan SETELAH training selesai — jangan dipakai sebagai alasan "kenapa training 22 kelas dulu bukan 19 dari awal" (anakronistik). Posisinya di proposal adalah sebagai pembanding metode di tinjauan pustaka, bukan justifikasi urutan training.

### Pengujian lapangan (field testing, di luar Colab)
- Setup: kamera HP + tripod, jarak tetap 25 cm dari citra wayang hasil CETAK (print, bukan wayang fisik asli — karena tidak ada akses; bukan foto dari layar — karena artefak moiré/glare).
- Hasil: **38 gambar diuji (114 data point dari 3x scan/gambar), 26/38 benar = ~68,4% akurasi** — turun signifikan dari 91% Colab.
- **Temuan 1 — cluster gagah/Kurawa terkonfirmasi ulang, lebih parah**: karna→kresna, duryudana→patih_sabrang/arjuna, dursasana→buta/sengkuni, sengkuni→petruk.
- **Temuan 2 — cluster BARU "ksatria halus"**: Arjuna, Puntadewa, Abimanyu, Nakula-Sadewa saling tertukar.
- **Temuan 3 — paradoks confidence**: prediksi BENAR dengan confidence sangat rendah (durna 31-37%, petruk 38-42%, togog 39-42%) vs prediksi SALAH dengan confidence sedang (karna→kresna 49%). Solusi: disambiguasi via disclaimer bertingkat & koreksi manual.
- **Temuan 4 — sensitivitas jarak/framing kamera**: Karna sempat salah terdeteksi Arjuna saat difoto jauh, benar saat didekatkan. Solusi: standarisasi jarak 25cm + tripod.

## PROGRES CHATBOT RAG & SISTEM INTEGRASI

### Temuan cross-class retrieval & Solusi Metadata Filter (SELESAI)
- **Masalah**: pure semantic search kadang mengambil chunk dari kelas lain yang mirip secara tekstual.
- **Solusi terimplementasi di `app.py`**: metadata filter kustom berbasis `lambda metadata: metadata.get("Karakter_CV") in target_karakter`.
- **Top-K Dinamis & Penanganan Varian Ejaan**:
  - Karakter Tunggal: `top_k = 5` (memastikan chunk sekunder seperti 'Anggota Keluarga' tidak terpotong akibat variasi ejaan seperti Anoman vs Hanoman).
  - Kasus `nakula_sadewa`: `top_k = 8` dengan pemetaan list `['nakula', 'sadewa']`.

### Kurasi knowledge base — kasus khusus
- **Wikipedia extraction**: strategi adaptif (≤6000/7500 karakter = ambil utuh, lebih = ringkasan + section terfilter, dengan fallback). Header section pakai `## Nama Section ##`.
- **Patih Sabrang**: bukan tokoh tunggal di Wikipedia (sempat salah kaprah ke "Pati Unus", sudah dikoreksi). Dikonfirmasi via blog Gubuk Wayang (Ki Wahyu Dunung, dalang bersertifikat): tokoh srambahan/arketipe, nama tidak baku. Draft manual sudah masuk CSV.
- **Buta**: golongan raksasa (bukan individu tunggal), mencakup Kumbakarna (tokoh tetap, dari Ramayana), Buto Raton, Buto Punuk, Brojodento (generik). Wikipedia "Raksasa" mayoritas tidak relevan (isi mitologi global), hanya 1 paragraf pembuka diambil. Draft manual sudah masuk CSV.
- **Nakula & Sadewa**: dipisah jadi 2 baris knowledge base terpisah meski CV tetap 1 kelas gabungan.

### Arsitektur Merger CV + RAG & UI Streamlit (SELESAI)

**Session State — 2 variabel terpisah (PENTING untuk AI yang mengerjakan kode):**
- `session_state.karakter_prediksi_cv` — menyimpan hasil prediksi mentah model CV & confidence score. **TIDAK PERNAH di-overwrite** sepanjang sesi, bahkan setelah user melakukan koreksi manual lewat dropdown/radio button. Fungsinya murni untuk **logging/analisis sesi** (seberapa sering user koreksi, kelas apa yang sering salah → bahan evaluasi Bab IV). Bukan untuk menjalankan logika chatbot.
- `session_state.karakter_aktif` — LIST nama karakter yang dikirim ke filter FAISS. Default = sama dengan prediksi CV. Bisa di-overwrite oleh koreksi manual user. Ini satu-satunya variabel yang dipakai oleh pipeline retrieval RAG.

**Fitur Sub-Fokus Nakula/Sadewa**: radio button UI (*Keduanya / Khusus Nakula / Khusus Sadewa*).

**Disclaimer Bertingkat Confidence CV:**
- Tinggi (>80%): Badge 🟢 tanpa keraguan.
- Sedang (50-80%): Badge 🟡 dengan disclaimer halus untuk memeriksa ulang.
- Rendah (<50%): Badge 🔴 dengan disclaimer tegas meminta pemeriksaan menu konfirmasi.
- Disclaimer ditempatkan DI DALAM welcome message chatbot (bukan popup terpisah) — conversational, tidak mengintimidasi.

**System Prompt Strict Grounding:**
- Instruksi ketat bahwa Gemini HANYA boleh menjawab berdasarkan potongan konteks ter-retrieve.
- Menolak berhalusinasi dari ingatan internet umum.
- Dilengkapi catatan sinonim ejaan wayang Jawa: Anoman=Hanoman, Bima=Werkudara, Gatotkaca=Gathotkaca/Tetuka, Kresna=Krisna, Durna=Drona.

**Limitasi desain yang perlu didokumentasikan di Bab IV:**
Sistem TIDAK mendukung perpindahan konteks karakter di tengah sesi hanya lewat ketikan teks (misal user mengetik "sekarang ceritakan tentang Arjuna" padahal yang di-scan adalah Karna). Retrieval FAISS tetap menggunakan `karakter_aktif` yang sudah di-set dari hasil scan, bukan dari nama yang disebut user di chat. Untuk pindah karakter, user harus: (1) klik tombol "Scan karakter lain" → scan ulang, ATAU (2) gunakan dropdown koreksi manual. Ini bukan kegagalan implementasi, tapi batasan desain yang disengaja — perlu dicantumkan jujur sebagai limitasi sistem di Bab IV.

## STATUS INTEGRASI TEKNIS (DEPLOYMENT-READY)
- Pipeline RAG standalone: **SUDAH SELESAI**
- Model CV standalone: **SUDAH SELESAI**
- Metadata filter (solusi cross-class): **SUDAH DIIMPLEMENTASIKAN & DIUJI (100% WORKING)**
- Integrasi CV + RAG dalam 1 aplikasi Streamlit (`app.py`): **SUDAH SELESAI & BERJALAN TERUJI LOKAL**
- Penanganan API Key Dual-Environment (`st.secrets` & `.env`) & Graceful Error Handling (`st.error`): **SUDAH SELESAI**
- Deployment ke Streamlit Community Cloud: **BELUM DILAKUKAN** (next step terakhir sisi teknis)

## PROGRES PROPOSAL TA / CONFERENCE INTERNAL KAMPUS

### Format
Template proposal internal ITB Asia Malang (7 bagian: Tema, Judul, Latar Belakang, Rumusan Masalah, Tujuan Penelitian, Metode Penelitian, Daftar Rujukan).

### Draft Proposal
- **Tema**: Integrasi Sistem Cerdas (Computer Vision dan Retrieval-Augmented Generation)
- **Judul**: "Pengenalan Karakter Wayang Jawa Menggunakan Arsitektur MobileNetV2 dan Generasi Teks Berbasis RAG"
- **Latar Belakang**: 3 paragraf. Paragraf 1 (fenomena masalah) SEDANG DIREVISI — objek diubah dari "masyarakat umum" jadi **"pelajar"** (revisi kalimat sudah dibuat, TAPI implikasi lanjutan ke rumusan masalah/metode pengujian usability/potensi penguatan argumen soal wayang sebagai muatan kurikulum lokal — BELUM dikonfirmasi final apakah diterapkan menyeluruh atau cuma di kalimat ini). Paragraf 2 (solusi CV) dan Paragraf 3 (solusi NLP/RAG) berisi sitasi yang ditulis mandiri mahasiswa (Maulana et al. 2025 — soal EfficientNetB2; Pratama & Avianto 2025; Bhuvaneswari & Varalakshmi 2026; Ramadhani et al. 2025; Tan et al. 2026; Zoupanos et al. 2022) — PENTING: belum diverifikasi AI, mahasiswa menyatakan yakin sumbernya ada.
- Ada pertanyaan terbuka soal frasa "image processing dan Computer Vision" vs "chatbot dan Computer Vision" di kalimat pembuka paragraf 2 — BELUM final, tergantung apakah paragraf tetap terpisah CV/NLP atau digabung jadi 1 kalimat pembuka mencakup keduanya.
- **Rumusan Masalah & Tujuan Penelitian** (masing-masing 3 poin, berkorespondensi 1:1): (1) implementasi & evaluasi MobileNetV2 transfer learning untuk 22 kelas dengan fokus pembahasan 19, (2) mekanisme integrasi filter metadata CV→FAISS, (3) efektivitas RAG berbasis Gemini dalam menekan halusinasi.

### Keputusan penting: 19 vs 22 kelas
- Dosen pembimbing meminta fokus pembahasan ke 19 karakter utama, mengeliminasi **Patih Sabrang, Durna, Cakil** — alasan dosen: fokus ke karakter utama/unggul secara naratif, bukan performa model.
- **Keputusan final: TIDAK retraining ulang** (tetap 22 kelas terlatih) — pembatasan 19 hanya di cakupan pembahasan laporan.
- Solusi penulisan: 2 tabel terpisah (Tabel 1: pembagian data aktual 22 kelas; Tabel 2: pemetaan 19 kelas fokus vs 3 kelas pendukung), disertai narasi eksplisit.
- **Narasi jawaban siap pakai untuk pertanyaan dosen "kenapa training 22 kelas dulu, bukan dari awal 19?"**: training 22 kelas dilakukan sebagai tahap eksplorasi awal SEBELUM konsultasi pembimbing; hasil evaluasi 22 kelas (confusion matrix) justru jadi bahan diskusi yang mengarah ke rekomendasi penyempitan fokus; retraining ulang tidak perlu karena model tetap valid. TIDAK menggunakan paper EfficientNetB2 sebagai alasan (anakronistik).

### Draft Metode Penelitian
3 paragraf naratif (data penelitian; alur CV; alur RAG + disclaimer bertingkat + metodologi pengujian lapangan 25cm). 2 flowchart dirancang (Tahap 1: pengembangan CV; Tahap 2: integrasi RAG & pengujian) — perlu digambar ulang di tools eksternal untuk dokumen Word. Rekomendasi contoh dataset: 1 gambar kolase 19 kelas, bukan 19 gambar terpisah.

### Belum dikerjakan di proposal
- Daftar Rujukan final (referensi asli sudah ditemukan AI: Sandler et al. 2018 MobileNetV2 original paper, beberapa paper RAG chatbot Indonesia — belum dikompilasi lengkap).
- Finalisasi & verifikasi mandiri seluruh sitasi mahasiswa.
- Keputusan final soal perubahan target audiens ke "pelajar" (dan implikasinya).
- Finalisasi frasa kalimat pembuka paragraf 2 (CV saja atau CV+chatbot).
- Kompilasi dokumen utuh format Word sesuai template resmi.

## PERSIAPAN UJIAN JUDUL

Dosen memberi bocoran 4 poin wajib bisa dijelaskan saat presentasi:

1. **Masalah**: pelajar kesulitan identifikasi visual wayang (kompleksitas ornamen, kemiripan antar tokoh kelompok sama) → sulit menggali nilai filosofis → menurunnya minat generasi muda. Lapis kedua: sekadar deteksi nama saja tidak cukup, chatbot LLM murni berisiko halusinasi untuk domain budaya lokal spesifik.
2. **Tujuan**: (1) model klasifikasi citra akurat, (2) chatbot faktual berbasis RAG, (3) integrasi keduanya jadi sistem utuh yang bisa dipakai langsung.
3. **Solusi/metode**: sistem 2 tahap — CV (MobileNetV2, 91% akurasi) mendeteksi karakter → hasil jadi konteks otomatis untuk chatbot RAG. Siap jawab "kenapa RAG bukan API biasa": karena API murni rentan halusinasi terutama untuk detail budaya lokal spesifik Jawa yang bisa beda dari versi Mahabharata/Ramayana asli India.
4. **Dataset**: 6.576 citra, 22 kelas, Kaggle, split 70:15:15 stratified. Knowledge base: ekstraksi adaptif Wikipedia + kurasi manual (Patih Sabrang, Buta).

Antisipasi pertanyaan susulan yang sudah disiapkan jawabannya:
- Kenapa fokus 19 dari 22 kelas → arahan pembimbing, fokus naratif bukan performa.
- Kenapa Durna dieliminasi padahal F1 1.0 → eliminasi berdasar signifikansi naratif, bukan kemampuan model.
- Apakah sudah diuji kondisi nyata → ya, field testing print+kamera 25cm, ditemukan penurunan akurasi 91%→68,4%, didokumentasikan sebagai limitasi.
- Bagaimana kalau confidence rendah → disclaimer bertingkat + koreksi manual, bukan blocking.
- Kenapa MobileNetV2 bukan EfficientNetB2 → trade-off akurasi vs efisiensi komputasi untuk deployment real-time terintegrasi RAG.

## NEXT STEPS (SISA PEKERJAAN YANG BELUM SELESAI)

### Sisi Teknis
1. Deploy ke **Streamlit Community Cloud** — push ke GitHub (pastikan `.env` di `.gitignore`), hubungkan ke Streamlit Cloud, setup secrets untuk API key, test link publik dari perangkat/browser lain.
2. Siapkan pesan pengantar untuk dosen saat mengirim link demo (catatan: loading pertama 30-60 detik karena cold start hosting gratis).

### Sisi Proposal & Akademis
3. Finalisasi keputusan target audiens "pelajar" (apakah menyeluruh ke rumusan masalah/metode atau hanya di kalimat latar belakang).
4. Finalisasi frasa kalimat pembuka paragraf 2 latar belakang (CV saja atau CV+chatbot).
5. Lengkapi Daftar Rujukan final — verifikasi mandiri semua sitasi mahasiswa, kompilasi referensi tambahan (Sandler et al. 2018, dll).
6. Kompilasi dokumen proposal utuh ke format Word sesuai template resmi kampus (termasuk: buat flowchart editable di draw.io/PowerPoint, buat kolase gambar dataset 19 kelas).
7. **Latihan presentasi ujian judul** — kuasai 4 poin bocoran dosen + antisipasi pertanyaan susulan di atas, target jawaban lisan 1-2 menit per poin, terdengar natural bukan hafalan.
