

# Real-Time Background Replacement dengan U-Net MobileNetV3

Proyek ini mengimplementasikan sistem penggantian latar belakang (background replacement) secara *real-time* untuk video conferencing berbasis web. Model ini dibangun menggunakan arsitektur **U-Net** dengan backbone **MobileNetV3-Large**, yang dirancang untuk efisiensi tinggi pada perangkat dengan sumber daya terbatas (CPU).

Aplikasi ini dikembangkan menggunakan **TensorFlow/Keras** untuk pelatihan dan **Streamlit + WebRTC** untuk antarmuka pengguna.

## Referensi Paper

Proyek ini mengacu pada metodologi yang dijelaskan dalam paper berikut, khususnya pada pendekatan *Lightweight U-Net MobileNet*:

> **Kiran Shahi, and Yongmin Li.** (2023). *Background Replacement in Video Conferencing*. International Journal of Network Dynamics and Intelligence.
>
> *Abstract:* Studi ini membandingkan U-Net standar dengan U-Net berbasis MobileNet untuk segmentasi semantik yang efisien guna memisahkan foreground dan background secara real-time.

---

## Antarmuka Pengguna (UI)

Berikut adalah tampilan antarmuka aplikasi saat dijalankan di browser:

![User Interface](image.png)

*Fitur UI meliputi: upload custom background, slider threshold, dan pilihan mode debug (Overlay Merah / Masker B&W).*

---

## Performa & Benchmark

Aplikasi telah diuji pada perangkat keras **Radeon 6600H**. Berikut adalah rata-rata FPS yang didapatkan berdasarkan mesin inferensi yang digunakan:

| Inference Engine | Rata-rata FPS | Keterangan |
| :--- | :---: | :--- |
| **TFLite (Quantized)** | **16 - 20 FPS** | ✅ **Rekomendasi** (CPU Optimized) |
| Keras (GPU) | 8 - 10 FPS | Menggunakan TensorFlow Direct |
| Keras (CPU) | 5 - 6 FPS | Paling lambat (Fallback) |

> **Catatan:** Model TFLite menggunakan kuantisasi default dan berjalan multi-threaded pada CPU, membuatnya jauh lebih cepat dibandingkan inferensi GPU standar untuk ukuran batch tunggal (1 frame).

---

## Arsitektur & Cara Kerja

Sistem ini menggunakan arsitektur **Encoder-Decoder** (U-Net).
1.  **Encoder (Backbone):** Menggunakan **MobileNetV3-Large** (pre-trained ImageNet) untuk mengekstrak fitur gambar secara progresif. Layer spesifik diambil untuk *skip connections*.
2.  **Decoder:** Menggunakan blok **Residual Decoder** kustom untuk merekonstruksi masker segmentasi dari fitur yang diekstrak, menggabungkan informasi spasial dari encoder (via skip connection) untuk detail tepian yang lebih baik.

### Diagram Arsitektur

![Model](model.png)

---

## Proses Training (`PCD_Akhir.ipynb`)

File notebook digunakan untuk melatih model dari awal hingga konversi ke TFLite.

1.  **Dataset:** Menggunakan dataset **Kaggle Person Segmentation** (nikhilroxtomar).
2.  **Preprocessing & Augmentasi:**
    *   Library: `Albumentations`.
    *   Teknik: *Coarse Dropout* (simulasi oklusi), *Channel Shuffle*, Rotasi, dan Flip. Ini penting agar model tahan terhadap gangguan visual.
3.  **Loss Function:** Hybrid Loss (**Binary Cross Entropy + Dice Loss**).
    *   *BCE:* Menjaga stabilitas klasifikasi pixel per pixel.
    *   *Dice:* Memaksimalkan overlap area masker (IOU).
4.  **Optimisasi Model:**
    *   Model disimpan dalam format `.keras`.
    *   Dikonversi ke `.tflite` dengan **Default Optimization (Quantization)** untuk mereduksi ukuran model (4x lebih kecil) dan mempercepat inferensi CPU.

---

## Penjelasan Program Aplikasi (`app.py`)

Aplikasi utama dibangun menggunakan **Streamlit**.

1.  **Inisialisasi:**
    *   Mengecek ketersediaan GPU dan library `ai-edge-litert` atau `tensorflow-lite`.
    *   Memuat model ke dalam cache (`@st.cache_resource`) agar tidak dimuat ulang setiap frame.
2.  **Video Processing (`PenggantiBackground` Class):**
    *   Menerima frame dari webcam via WebRTC.
    *   **Resize:** Mengubah ukuran frame ke 256x256 pixel (input model).
    *   **Inferensi:**
        *   Jika **TFLite**: Menggunakan `Interpreter` dengan *multi-threading*.
        *   Jika **Keras**: Menggunakan `model.predict` standar.
    *   **Post-Processing:** Resize output mask kembali ke ukuran asli webcam, melakukan thresholding, dan menggabungkan (compositing) dengan background baru.
3.  **WebRTC:** Menangani streaming video agar tetap berjalan lancar di browser tanpa mengirim gambar ke server backend (pemrosesan lokal/klien jika memungkinkan, atau server-side processing yang efisien).

---

## Known Issues (Bug)

**Resolusi Kamera di Browser (Chrome/Chromium)**
Pada konteks yang tidak aman (HTTP, bukan HTTPS), Chrome secara paksa membatasi stream kamera.
> *On insecure contexts Chrome/Chromium artificially caps every camera stream at 640 × 480 (and often falls back to 360p) for privacy/security reasons.*
>
> Jika Anda menjalankan di `http://localhost:8501`, Anda mungkin melihat kualitas video turun atau terpotong. Solusinya adalah menjalankan aplikasi pada konteks HTTPS atau mengabaikan penurunan resolusi ini saat development.

---

## Cara Menjalankan

### 1. Persiapan Environment

Disarankan menggunakan Python 3.9 - 3.11.

**A. Requirement untuk Training (Jupyter Notebook)**
Install library berikut jika ingin melatih ulang model:
```bash
pip install tensorflow opencv-python matplotlib albumentations scikit-learn
```

**B. Requirement untuk Menjalankan Aplikasi (App)**
Buat file `requirements.txt` dengan isi:
```text
streamlit
streamlit-webrtc
opencv-python-headless
tensorflow
ai-edge-litert
numpy
Pillow
av
```
Lalu install:
```bash
pip install -r requirements.txt
```

*(Opsional: Jika ingin performa TFLite maksimal tanpa install full TensorFlow, gunakan `tflite-runtime` atau `ai-edge-litert`)*.

### 2. Menjalankan Aplikasi
Pastikan file `best_model_fixed.keras` dan `model_quantized.tflite` berada dalam satu folder dengan `app.py`.

Jalankan perintah:
```bash
streamlit run app.py
```

Akses aplikasi melalui browser di alamat yang muncul di terminal (biasanya `http://localhost:8501`).

---

### Struktur Folder
```text
|-- PCD_Akhir.ipynb         # Notebook pelatihan & konversi model
|-- app.py                  # Aplikasi utama Streamlit
|-- best_model_fixed.keras  # Model hasil training (Full Precision)
|-- model_quantized.tflite  # Model hasil optimasi (Untuk App)
|-- image.png               # Screenshot UI
`-- __pycache__
```