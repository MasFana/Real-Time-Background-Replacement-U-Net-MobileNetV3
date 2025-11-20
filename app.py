import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import av
import time
import os

# ==========================================
# 0. PENGATURAN HALAMAN & CEK LIBRARY
# ==========================================
st.set_page_config(
    page_title="U-NET MobileNetV3 - Realtime Background Replacement", 
    layout="wide", 
    page_icon="🎬"
)

# Cek ketersediaan library LiteRT 
try:
    from ai_edge_litert.interpreter import Interpreter
    LITERT_AVAILABLE = True
    STATUS_MESIN = "Menggunakan Akselerasi LiteRT"
except ImportError:
    LITERT_AVAILABLE = False
    STATUS_MESIN = "Menggunakan TensorFlow Lite Legacy"
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.interpreter import Interpreter

# Cek Ketersediaan GPU Fisik
GPUS = tf.config.list_physical_devices('GPU')
GPU_AVAILABLE = len(GPUS) > 0

# ==========================================
# 1. KONFIGURASI & CACHE MODEL
# ==========================================
IMG_SIZE = 256
KERAS_MODEL_PATH = 'best_model_fixed.keras' 
TFLITE_MODEL_PATH = 'model_quantized.tflite'

# Fungsi dummy custom objects
def dice_coef(y_true, y_pred): return 0.0
def bce_dice_loss(y_true, y_pred): return 0.0

@st.cache_resource
def muat_model_keras():
    """Memuat model Keras."""
    if not os.path.exists(KERAS_MODEL_PATH): return None
    try:
        # Set memory growth agar GPU tidak langsung penuh (jika ada)
        for gpu in GPUS:
            tf.config.experimental.set_memory_growth(gpu, True)
            
        return tf.keras.models.load_model(
            KERAS_MODEL_PATH, 
            custom_objects={'dice_coef': dice_coef, 'bce_dice_loss': bce_dice_loss},
            compile=False 
        )
    except Exception as e:
        st.error(f"Gagal memuat Keras: {e}")
        return None

@st.cache_resource
def muat_model_tflite():
    """Memuat model TFLite dengan Optimalisasi Thread."""
    if not os.path.exists(TFLITE_MODEL_PATH): return None
    try:
        # OPTIMALISASI: Gunakan jumlah core CPU yang tersedia (max 4 agar stabil di cloud/laptop)
        cpu_count = os.cpu_count() if os.cpu_count() else 4
        threads = min(cpu_count, 4) 
        
        interpreter = Interpreter(model_path=TFLITE_MODEL_PATH, num_threads=threads)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Gagal memuat TFLite: {e}")
        return None

# ==========================================
# 2. PEMROSESAN VIDEO 
# ==========================================
class PenggantiBackground(VideoProcessorBase):
    def __init__(self):
        self.threshold = 0.5
        self.bg_image = None
        self.tipe_model = "TFLite" # Default
        self.mode_tampilan = "Ganti Background"
        
        self.waktu_sebelumnya = 0
        self.fps = 0
        
        self.model_keras = None
        self.interpreter_tflite = None
        self.input_details = None
        self.output_details = None

    def update_pengaturan(self, threshold, bg_image, tipe_model, mode_tampilan, model_keras, interpreter_tflite):
        self.threshold = threshold
        self.tipe_model = tipe_model
        self.mode_tampilan = mode_tampilan
        
        if bg_image is not None:
            img_array = np.array(bg_image.convert('RGB'))
            self.bg_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            self.bg_image = None

        self.model_keras = model_keras
        self.interpreter_tflite = interpreter_tflite
        
        if self.interpreter_tflite:
            self.input_details = self.interpreter_tflite.get_input_details()
            self.output_details = self.interpreter_tflite.get_output_details()

    def prediksi_masker(self, img_input):
        """
        Logika Pemilihan Model (TFLite / Keras CPU / Keras GPU)
        """
        # 1. MODE TFLITE
        if self.tipe_model == "TFLite" and self.interpreter_tflite:
            self.interpreter_tflite.set_tensor(self.input_details[0]['index'], img_input)
            self.interpreter_tflite.invoke()
            return self.interpreter_tflite.get_tensor(self.output_details[0]['index'])[0]
        
        # 2. MODE KERAS (CPU & GPU)
        elif "Keras" in self.tipe_model and self.model_keras:
            device_name = '/GPU:0' if "GPU" in self.tipe_model and GPU_AVAILABLE else '/CPU:0'
            
            with tf.device(device_name):
                return self.model_keras({"input_layer":img_input}, training=False)[0].numpy()
                
        return np.zeros((IMG_SIZE, IMG_SIZE, 1))

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        # Hitung FPS
        waktu_sekarang = time.time()
        selisih_waktu = waktu_sekarang - self.waktu_sebelumnya
        if selisih_waktu > 0: self.fps = 1 / selisih_waktu
        self.waktu_sebelumnya = waktu_sekarang

        # Resize Input
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        img_input = np.expand_dims(img_resized.astype(np.float32), axis=0)
        
        # PREDIKSI
        mask = self.prediksi_masker(img_input)

        # Resize Output & Thresholding
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_binary = (mask_resized > self.threshold).astype(np.float32)

        # Komposisi
        hasil_akhir = None
        
        if self.mode_tampilan == "Lihat Masker (B&W)":
            hasil_akhir = cv2.cvtColor((mask_binary * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            
        elif self.mode_tampilan == "Cek Seleksi (Overlay Merah)":
            overlay_merah = np.zeros_like(img)
            overlay_merah[:, :, 2] = 255 
            mask_3d = np.stack((mask_binary,)*3, axis=-1)
            hasil_akhir = np.where(mask_3d > 0, cv2.addWeighted(img, 0.7, overlay_merah, 0.3, 0), img)

        else: 
            mask_3d = np.stack((mask_binary,)*3, axis=-1)
            if self.bg_image is not None:
                if self.bg_image.shape[:2] != (h, w):
                    bg_siap = cv2.resize(self.bg_image, (w, h))
                else:
                    bg_siap = self.bg_image
            else:
                bg_siap = np.zeros_like(img)
                bg_siap[:] = (0, 255, 0) 

            # Menggunakan tipe data uint8 langsung setelah perhitungan selesai untuk kecepatan
            hasil_akhir = (img * mask_3d + bg_siap * (1.0 - mask_3d)).astype(np.uint8)

        # Info Mode + Resolusi
        teks_mode = f"Mode: {self.tipe_model}"
        teks_info = f"FPS: {int(self.fps)} | {w}x{h} | {teks_mode}"
        cv2.putText(
            hasil_akhir,
            teks_info,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        return av.VideoFrame.from_ndarray(hasil_akhir, format="bgr24")

# ==========================================
# 3. TAMPILAN ANTARMUKA (UI)
# ==========================================
st.title("U-NET MobileNetV3 - Realtime Background Replacement")

col_video, col_settings = st.columns([2, 1])

with col_settings:
    st.header("Pengaturan")
    
    file_bg = st.file_uploader("Pilih Background", type=["jpg", "png", "jpeg"])
    st.write("---")
    
    pilihan_mode = st.radio(
        "Tampilan", 
        ["Ganti Background", "Cek Seleksi (Overlay Merah)", "Lihat Masker (B&W)"]
    )
    
    st.write("---")
    nilai_threshold = st.slider("Threshold", 0.1, 0.9, 0.5)

    # -------------------------------------------------
    # PENGATURAN MODE MESIN (DIPERBARUI)
    # -------------------------------------------------
    with st.expander("Pilih Mesin AI (Performance)", expanded=True):
        opsi_mesin = ["TFLite"]
        opsi_mesin.append("Keras (CPU)")
        
        label_gpu = "Keras (GPU)"
        if GPU_AVAILABLE:
            label_gpu += " ✅ Ready"
        else:
            label_gpu += " ❌ Not Detected"
        opsi_mesin.append(label_gpu)
        
        pilihan_mesin = st.radio("Jalankan Model Di:", opsi_mesin)

# ==========================================
# 4. LOGIKA WEBRTC
# ==========================================

model_keras = muat_model_keras()
model_tflite = muat_model_tflite()

# Normalisasi string pilihan user ke kode internal
if "TFLite" in pilihan_mesin:
    nama_tipe_model = "TFLite"
elif "CPU" in pilihan_mesin:
    nama_tipe_model = "Keras CPU"
else:
    nama_tipe_model = "Keras GPU"

# Cek kelengkapan
siap_jalan = False
if nama_tipe_model == "TFLite" and model_tflite:
    siap_jalan = True
elif "Keras" in nama_tipe_model and model_keras:
    siap_jalan = True
    if "GPU" in nama_tipe_model and not GPU_AVAILABLE:
        st.warning("⚠️ GPU tidak terdeteksi! Sistem akan otomatis fallback ke CPU.")

if siap_jalan:
    gambar_latar = Image.open(file_bg) if file_bg else None
    
    constraints_otomatis = {
        "audio": False, 
        "video": {
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
            "facingMode": "user"
        }
    }

    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    with col_video:
        ctx = webrtc_streamer(
            key="mas-fana-sigma",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            media_stream_constraints=constraints_otomatis, 
            video_processor_factory=PenggantiBackground,
            async_processing=True,
        )

    if ctx.video_processor:
        ctx.video_processor.update_pengaturan(
            threshold=nilai_threshold,
            bg_image=gambar_latar,
            tipe_model=nama_tipe_model,
            mode_tampilan=pilihan_mode,
            model_keras=model_keras,
            interpreter_tflite=model_tflite
        )

else:
    st.error("⚠️ File Model (Keras/TFLite) tidak ditemukan.")