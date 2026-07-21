import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Ensure backend directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from plate_finder import PlateFinder
from ocr_engine import OCREngine

# Page config
st.set_page_config(
    page_title="LPR-Vision LAB EDITION",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load engines once
@st.cache_resource
def load_lpr_engines():
    finder = PlateFinder()
    ocr = OCREngine()
    return finder, ocr

finder, ocr_engine = load_lpr_engines()

# Session State
if "whitelist" not in st.session_state:
    st.session_state["whitelist"] = ["A5998", "KL01CB1234", "MH12DE1433", "DL3CCE0001", "HR98TC0048"]

if "history" not in st.session_state:
    st.session_state["history"] = []

# Styling
st.markdown("""
<style>
    .stApp { background-color: #F8FAFC; color: #1E293B; }
    
    .top-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #FFFFFF;
        padding: 14px 24px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 20px;
        border: 1px solid #E2E8F0;
    }
    .brand-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #0F172A;
    }
    .badge-lab {
        background-color: #E0F2FE;
        color: #0284C7;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 12px;
        margin-left: 8px;
    }
    .status-badge-green {
        color: #16A34A;
        font-weight: 600;
        font-size: 0.85rem;
    }

    .hud-box {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }
    
    .license-plate-graphic {
        border: 3px solid #0F172A;
        border-radius: 8px;
        background: #FFFFFF;
        display: flex;
        align-items: center;
        padding: 6px 14px;
        margin: 15px 0;
    }
    .ind-strip {
        background: #003399;
        color: white;
        font-weight: 900;
        font-size: 0.65rem;
        padding: 8px 5px;
        border-radius: 4px;
        text-align: center;
        line-height: 1;
        margin-right: 12px;
    }
    .plate-text {
        font-family: 'Courier New', monospace;
        font-weight: 900;
        font-size: 2rem;
        color: #0F172A;
        letter-spacing: 4px;
        width: 100%;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Top Bar
st.markdown("""
<div class="top-header">
    <div class="brand-title">
        🚗 LPR-Vision <span class="badge-lab">LAB EDITION</span>
    </div>
    <div>
        <span class="status-badge-green">🟢 Backend: Connected</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        <span class="status-badge-green">🟢 OCR Engine: Ready</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎛️ CV Parameter Tuner")
ocr_mode = st.sidebar.selectbox("Recognition Engine", ["Neural (EasyOCR)", "Classic CV (GeG Contours)"])
gaussian_blur = st.sidebar.select_slider("Gaussian Blur Kernel", options=[3, 5, 7, 9, 11], value=7)
sobel_size = st.sidebar.select_slider("Sobel X Kernel Size", options=[3, 5, 7], value=3)
morph_w = st.sidebar.slider("Morphology Close Width", 10, 40, 22)
morph_h = st.sidebar.slider("Morphology Close Height", 1, 10, 3)
min_area = st.sidebar.number_input("Min Plate Area (px)", value=500, step=100)
max_area = st.sidebar.number_input("Max Plate Area (px)", value=30000, step=1000)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Whitelist Manager")
new_plate = st.sidebar.text_input("Add Plate Number:", placeholder="E.G. DL3CAG1234").strip().upper()
if st.sidebar.button("➕ Add to Whitelist") and new_plate:
    if new_plate not in st.session_state["whitelist"]:
        st.session_state["whitelist"].append(new_plate)
        st.sidebar.success(f"Added {new_plate}")

st.sidebar.caption("Authorized Whitelist:")
for p in st.session_state["whitelist"]:
    st.sidebar.code(p)

# Main UI Tabs
tab1, tab2, tab3 = st.tabs(["📷 Live Detector", "📊 Visual CV Pipeline", "📜 Audit Logs"])

with tab1:
    col_input, col_hud = st.columns([1.3, 1.0])
    
    input_bgr = None
    
    with col_input:
        st.markdown('<div class="hud-box">', unsafe_allow_html=True)
        st.subheader("Media Input")
        
        mode = st.radio("Choose Input Source:", ["Upload Vehicle Image", "Live Webcam", "Sample Video"], horizontal=True)
        
        if mode == "Upload Vehicle Image":
            uploaded = st.file_uploader("Choose an image file (JPG, PNG)", type=["jpg", "jpeg", "png"])
            if uploaded:
                img = Image.open(uploaded)
                input_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                st.image(img, use_column_width=True)
            else:
                sample_p = os.path.join(BACKEND_DIR, "static", "debug_original.jpg")
                if os.path.exists(sample_p):
                    input_bgr = cv2.imread(sample_p)
                    st.image(cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB), caption="Default Sample Vehicle", use_column_width=True)
                    
        elif mode == "Live Webcam":
            snap = st.camera_input("Take a photo")
            if snap:
                img = Image.open(snap)
                input_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        elif mode == "Sample Video":
            vid_p = os.path.join(BASE_DIR, "test_car.mp4")
            if os.path.exists(vid_p):
                st.video(vid_p)
                if st.button("▶️ Extract & Process Frame from Video"):
                    cap = cv2.VideoCapture(vid_p)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        input_bgr = frame

        st.markdown("<br>", unsafe_allow_html=True)
        # EXPLICIT DETECT BUTTON
        btn_detect = st.button("🔍 DETECT LICENSE PLATE", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_hud:
        st.markdown('<div class="hud-box">', unsafe_allow_html=True)
        st.subheader("💡 Real-time HUD")
        
        plate_text = "------"
        conf_val = 0.0
        verdict = "NO PLATE"
        plate_crop = None
        
        if input_bgr is not None and (btn_detect or mode == "Sample Video"):
            finder.min_area = min_area
            finder.max_area = max_area
            finder.blur_kernel = gaussian_blur
            finder.sobel_kernel = sobel_size
            finder.morph_w = morph_w
            finder.morph_h = morph_h
            
            with st.spinner("Analyzing image contours & neural OCR..."):
                plates, annotated = finder.find_candidates(input_bgr)
                mode_name = "neural" if "Neural" in ocr_mode else "classic"
                
                if plates:
                    for p in plates[:3]:
                        txt, c, _ = ocr_engine.recognize_plate(p, mode=mode_name)
                        if txt and len(txt) >= 3:
                            plate_text = txt
                            conf_val = c
                            plate_crop = p
                            break
                            
                if plate_text == "------" and ocr_engine.reader is not None:
                    results = ocr_engine.reader.readtext(input_bgr)
                    for bbox, txt, c in results:
                        clean = "".join(e for e in txt if e.isalnum()).upper()
                        if len(clean) >= 3:
                            plate_text = clean
                            conf_val = float(c)
                            break
                            
                if plate_text != "------":
                    verdict = "AUTHORIZED" if plate_text in st.session_state["whitelist"] else "UNAUTHORIZED"
                    st.session_state["history"].insert(0, {
                        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "Plate Number": plate_text,
                        "Confidence": f"{conf_val*100:.1f}%",
                        "Verdict": verdict
                    })

        # Plate Box Display
        st.markdown(f"""
        <div class="license-plate-graphic">
            <div class="ind-strip">I<br>N<br>D</div>
            <div class="plate-text">{plate_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**System Verdict:**")
            if verdict == "AUTHORIZED":
                st.success("✅ AUTHORIZED")
            elif verdict == "UNAUTHORIZED":
                st.error("⛔ UNAUTHORIZED")
            else:
                st.info("⚠️ NO PLATE DETECTED")
                
        with c2:
            st.write("**Confidence:**")
            st.markdown(f"### {conf_val*100:.1f}%")

        st.write("**Segmented Crop:**")
        if plate_crop is not None:
            st.image(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB), width=220)
        else:
            st.info("No crop segment available")

        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.subheader("📊 Visual Computer Vision Pipeline")
    if input_bgr is not None:
        gray = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (gaussian_blur, gaussian_blur), 0)
        sobelx = cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=sobel_size)
        _, thresh = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.caption("1. Grayscale")
        c1.image(gray, use_column_width=True)
        c2.caption("2. Gaussian Blur")
        c2.image(blur, use_column_width=True)
        c3.caption("3. Sobel Vertical Edges")
        c3.image(thresh, use_column_width=True)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_w, morph_h))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        c4.caption("4. Morphological Close")
        c4.image(morph, use_column_width=True)

with tab3:
    st.subheader("📜 System Audit Logs")
    if st.session_state["history"]:
        st.dataframe(pd.DataFrame(st.session_state["history"]), use_container_width=True)
    else:
        st.info("No audit logs recorded yet.")
