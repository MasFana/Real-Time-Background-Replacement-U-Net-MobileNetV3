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
except ImportError:
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

def dice_coef(y_true, y_pred): return 0.0
def bce_dice_loss(y_true, y_pred): return 0.0

@st.cache_resource
def muat_model_keras():
    if not os.path.exists(KERAS_MODEL_PATH): return None
    try:
        for gpu in GPUS: tf.config.experimental.set_memory_growth(gpu, True)
        return tf.keras.models.load_model(
            KERAS_MODEL_PATH, 
            custom_objects={'dice_coef': dice_coef, 'bce_dice_loss': bce_dice_loss},
            compile=False 
        )
    except Exception as e: return None

@st.cache_resource
def muat_model_tflite():
    if not os.path.exists(TFLITE_MODEL_PATH): return None
    try:
        threads = min(os.cpu_count() or 4, 4) 
        interpreter = Interpreter(model_path=TFLITE_MODEL_PATH, num_threads=threads)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e: return None

# ==========================================
# 2. PEMROSESAN VIDEO 
# ==========================================
class PenggantiBackground(VideoProcessorBase):
    def __init__(self):
        # Setting Default
        self.threshold = 0.5
        self.bg_image = None
        self.tipe_model = "TFLite"
        self.mode_tampilan = "Ganti Background"
        
        # Setting Preprocessing
        self.denoise_type = "None"
        self.use_clahe = False
        self.clahe_clip = 2.0
        self.show_preprocess_view = False
        
        # Init Objects
        self.clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.waktu_sebelumnya = 0
        self.fps = 0
        
        self.model_keras = None
        self.interpreter_tflite = None
        self.input_details = None
        self.output_details = None

    def update_pengaturan(self, params):
        """Menerima dictionary parameter dari Streamlit UI"""
        self.threshold = params['threshold']
        self.tipe_model = params['tipe_model']
        self.mode_tampilan = params['mode_tampilan']
        self.denoise_type = params['denoise_type']
        self.use_clahe = params['use_clahe']
        self.show_preprocess_view = params['show_preprocess_view']
        
        # Update CLAHE Clip
        if self.clahe_clip != params['clahe_clip']:
            self.clahe_clip = params['clahe_clip']
            self.clahe_obj = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))

        # Update Background
        bg_file = params['bg_image']
        if bg_file is not None:
            img_array = np.array(bg_file.convert('RGB'))
            self.bg_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            self.bg_image = None

        # Update Model References
        self.model_keras = params['model_keras']
        self.interpreter_tflite = params['model_tflite']
        
        if self.interpreter_tflite:
            self.input_details = self.interpreter_tflite.get_input_details()
            self.output_details = self.interpreter_tflite.get_output_details()

    def terapkan_clahe(self, img):
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = self.clahe_obj.apply(l)
            lab_updated = cv2.merge((l_clahe, a, b))
            return cv2.cvtColor(lab_updated, cv2.COLOR_LAB2BGR)
        except Exception:
            return img

    def prediksi_masker(self, img_input):
        if self.tipe_model == "TFLite" and self.interpreter_tflite:
            self.interpreter_tflite.set_tensor(self.input_details[0]['index'], img_input)
            self.interpreter_tflite.invoke()
            return self.interpreter_tflite.get_tensor(self.output_details[0]['index'])[0]
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

        # ==========================================
        # 1. PIPELINE PRE-PROCESSING (UNTUK AI)
        # ==========================================
        img_proc = img.copy()

        # A. FILTER NOISE
        if self.denoise_type == "Median Blur":
            img_proc = cv2.medianBlur(img_proc, 5)
        elif self.denoise_type == "Bilateral Filter":
            img_proc = cv2.bilateralFilter(img_proc, 9, 75, 75)
        
        # B. ENHANCE CONTRAST (CLAHE)
        if self.use_clahe:
            img_proc = self.terapkan_clahe(img_proc)

        # C. LOGIKA TAMPILAN (DEBUG VIEW)
        if self.show_preprocess_view:
            status = f"FPS: {int(self.fps)} | {w}x{h} | DEBUG: {self.denoise_type} + CLAHE"
            cv2.putText(img_proc, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return av.VideoFrame.from_ndarray(img_proc, format="bgr24")

        # ==========================================
        # 2. INFERENCE AI
        # ==========================================
        img_resized = cv2.resize(img_proc, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        img_input = np.expand_dims(img_resized.astype(np.float32), axis=0)
        
        mask = self.prediksi_masker(img_input)

        # ==========================================
        # 3. POST-PROCESSING & COMPOSITING
        # ==========================================
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_binary = (mask_resized > self.threshold).astype(np.uint8)

        # Morphological Closing
        kernel = np.ones((5,5), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
        mask_float = mask_cleaned.astype(np.float32)

        hasil_akhir = None
        
        if self.mode_tampilan == "Lihat Masker (B&W)":
            hasil_akhir = cv2.cvtColor((mask_cleaned * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            
        elif self.mode_tampilan == "Cek Seleksi (Overlay Merah)":
            overlay_merah = np.zeros_like(img)
            overlay_merah[:, :, 2] = 255 
            mask_3d = np.stack((mask_float,)*3, axis=-1)
            hasil_akhir = np.where(mask_3d > 0, img, cv2.addWeighted(img, 0.7, overlay_merah, 0.3, 0))

        else: 
            # GANTI BACKGROUND
            mask_3d = np.stack((mask_float,)*3, axis=-1)
            
            if self.bg_image is not None:
                if self.bg_image.shape[:2] != (h, w):
                    bg_siap = cv2.resize(self.bg_image, (w, h))
                else:
                    bg_siap = self.bg_image
            else:
                bg_siap = np.zeros_like(img)
                bg_siap[:] = (0, 255, 0) 

            hasil_akhir = (img * mask_3d + bg_siap * (1.0 - mask_3d)).astype(np.uint8)

        # ------------------------------------------------------------------
        # MENAMPILKAN INFO TEKS (FPS & RESOLUSI)
        # ------------------------------------------------------------------
        teks_info = f"FPS: {int(self.fps)} | Resolusi: {w}x{h}"
        cv2.putText(hasil_akhir, teks_info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(hasil_akhir, format="bgr24")

# ==========================================
# 3. TAMPILAN ANTARMUKA (UI)
# ==========================================
st.title("U-NET MobileNetV3 - Realtime Background Replacement")

col_video, col_settings = st.columns([2, 1])

# --- LOGIKA UI ---
with col_settings:
    st.header("Pengaturan")
    
    # 1. Background & Mode
    file_bg = st.file_uploader("Pilih Background", type=["jpg", "png", "jpeg"])
    pilihan_mode = st.radio("Tampilan", ["Ganti Background", "Cek Seleksi (Overlay Merah)", "Lihat Masker (B&W)"])
    nilai_threshold = st.slider("Threshold AI", 0.1, 0.9, 0.5)

    # 2. Pre-Processing Section
    with st.expander("Advanced Pre-Processing (Filter Noise & Cahaya)", expanded=True):
        st.caption("Memproses gambar input agar lebih mudah dikenali AI.")
        
        # Denoise
        pilih_denoise = st.selectbox(
            "Filter Noise (Bintik)", 
            ["None", "Median Blur", "Bilateral Filter"],
            help="Median: Bagus untuk bintik pasir. Bilateral: Halus tapi berat."
        )
        
        # CLAHE
        pakai_clahe = st.checkbox("Aktifkan CLAHE (Kontras)", value=False)
        nilai_clahe = st.slider("Kekuatan Kontras", 1.0, 10.0, 2.0, 0.5, disabled=not pakai_clahe)
        
        # Debug View
        st.markdown("---")
        lihat_preprocess = st.checkbox("Lihat Input Model (Debug)", value=False, 
                                      help="Lihat gambar yang sebenarnya diproses oleh AI (Blur + High Contrast)")

    # 3. Model Engine
    with st.expander("Pilih Mesin AI", expanded=False):
        opsi_mesin = ["TFLite", "Keras (CPU)"]
        if GPU_AVAILABLE: opsi_mesin.append("Keras (GPU) ✅")
        pilihan_mesin = st.radio("Engine:", opsi_mesin)

# ==========================================
# 4. LOGIKA WEBRTC
# ==========================================
model_keras = muat_model_keras()
model_tflite = muat_model_tflite()

# Parsing Pilihan Mesin
nama_tipe_model = "TFLite"
if "Keras" in pilihan_mesin:
    nama_tipe_model = "Keras GPU" if "GPU" in pilihan_mesin else "Keras CPU"

# Validasi Model
siap_jalan = False
if nama_tipe_model == "TFLite" and model_tflite: siap_jalan = True
elif "Keras" in nama_tipe_model and model_keras: siap_jalan = True

if siap_jalan:
    gambar_latar = Image.open(file_bg) if file_bg else None
    
    constraints_otomatis = {
        "audio": False, 
        "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}, "facingMode": "user"}
    }
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    with col_video:
        ctx = webrtc_streamer(
            key="stream-clahe",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            media_stream_constraints=constraints_otomatis, 
            video_processor_factory=PenggantiBackground,
            async_processing=True,
        )

    # Kirim Parameter Realtime ke Processor
    if ctx.video_processor:
        params = {
            'threshold': nilai_threshold,
            'bg_image': gambar_latar,
            'tipe_model': nama_tipe_model,
            'mode_tampilan': pilihan_mode,
            'model_keras': model_keras,
            'model_tflite': model_tflite,
            # Parameter Baru
            'denoise_type': pilih_denoise,
            'use_clahe': pakai_clahe,
            'clahe_clip': nilai_clahe,
            'show_preprocess_view': lihat_preprocess
        }
        ctx.video_processor.update_pengaturan(params)

else:
    st.error("⚠️ File Model tidak ditemukan.")