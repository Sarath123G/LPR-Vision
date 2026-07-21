import os
import cv2
import json
import uuid
import base64
import numpy as np
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
import io

from plate_finder import PlateFinder, to_base64
from ocr_engine import OCREngine

# Initialize FastAPI App
app = FastAPI(title="LPR-Vision Backend")

# Setup CORS so the frontend can interact from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories for saving artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
PLATES_DIR = os.path.join(STATIC_DIR, "plates")
VIDEOS_DIR = os.path.join(STATIC_DIR, "videos")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(PLATES_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

# Mount Static Files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Database File Paths
WHITELIST_PATH = os.path.join(BASE_DIR, "whitelist.json")
HISTORY_PATH = os.path.join(BASE_DIR, "history.json")

# Load or Initialize Whitelist
if os.path.exists(WHITELIST_PATH):
    try:
        with open(WHITELIST_PATH, "r") as f:
            whitelist = json.load(f)
    except Exception:
        whitelist = ["DL3CAG1234", "MH12DE1433", "KA05NB9999", "CAR123", "MH12DE1435"]
else:
    whitelist = ["DL3CAG1234", "MH12DE1433", "KA05NB9999", "CAR123", "MH12DE1435"]
    with open(WHITELIST_PATH, "w") as f:
        json.dump(whitelist, f)

# Load or Initialize History
if os.path.exists(HISTORY_PATH):
    try:
        with open(HISTORY_PATH, "r") as f:
            history = json.load(f)
    except Exception:
        history = []
else:
    history = []
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f)

# Initialize OCR Engine
ocr_engine = OCREngine()

# Pydantic Schemas
class WebcamFrame(BaseModel):
    image: str # Base64 data URL
    engine: str # "classic" (GfG) or "ai" (EasyOCR whole plate)
    min_area: int = 1000
    max_area: int = 30000
    min_ratio: float = 2.5
    max_ratio: float = 7.0
    blur_kernel: int = 7
    sobel_kernel: int = 3
    morph_w: int = 22
    morph_h: int = 3

class WhitelistItem(BaseModel):
    plate: str

# Helper to save detection to history
def add_history_record(plate_text, cropped_plate_img, confidence):
    global history
    confidence_val = float(confidence)
    plate_text_clean = plate_text.replace(" ", "").upper()
    
    # Check Whitelist status
    is_authorized = plate_text_clean in [p.replace(" ", "").upper() for p in whitelist]
    
    # Save cropped image file
    filename = f"plate_{uuid.uuid4().hex[:8]}.jpg"
    filepath = os.path.join(PLATES_DIR, filename)
    
    cv2.imwrite(filepath, cropped_plate_img)
    
    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "plate": plate_text.upper(),
        "confidence": round(confidence_val * 100, 1) if confidence_val <= 1.0 else round(confidence_val, 1),
        "status": "Authorized" if is_authorized else "Unauthorized",
        "image_url": f"/static/plates/{filename}"
    }
    
    # Insert at beginning
    history.insert(0, record)
    
    # Write to file
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f)
        
    return record


@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "LPR-Vision API is running successfully!"}

@app.post("/api/process-image")
async def process_image(
    image: UploadFile = File(...),
    engine: str = Form("classic"), # "classic" (segmented chars) or "ai" (whole plate)
    min_area: int = Form(500),
    max_area: int = Form(30000),
    min_ratio: float = Form(1.5),
    max_ratio: float = Form(7.0),
    blur_kernel: int = Form(7),
    sobel_kernel: int = Form(3),
    morph_w: int = Form(22),
    morph_h: int = Form(3),
):
    try:
        # Read uploaded image directly using OpenCV
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            try:
                pil_img = Image.open(io.BytesIO(contents))
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception:
                pass
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file format.")
            
        # Save original image for debugging
        cv2.imwrite(os.path.join(STATIC_DIR, "debug_original.jpg"), img)

        # Initialize Plate Finder
        finder = PlateFinder(
            min_plate_area=min_area,
            max_plate_area=max_area,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            blur_kernel=blur_kernel,
            sobel_kernel=sobel_kernel,
            morph_w=morph_w,
            morph_h=morph_h
        )
        
        # Run CV Pipeline
        cv_res = finder.process_frame(img)
        
        # Perform OCR if plate is found
        raw_candidates = []
        for i, cropped_plate in enumerate(cv_res["plates"]):
            coords = cv_res["coordinates"][i]
            
            # Perform neural / segmented OCR
            plate_text, conf = ocr_engine.recognize_plate(cropped_plate, use_ai=True)
            if not plate_text and engine == "classic":
                segmented_chars = cv_res["characters"][i]
                plate_text, conf = ocr_engine.recognize_characters(segmented_chars)
                
            conf = float(conf)
            clean_t = plate_text.replace(" ", "").strip()
            if len(clean_t) >= 3:
                raw_candidates.append({
                    "text": plate_text,
                    "clean_text": clean_t,
                    "confidence": conf,
                    "box": coords,
                    "cropped_plate": cropped_plate
                })
        
        # Whole-Image Neural Fallback: if no valid candidates found or to supplement contour candidates
        if (not raw_candidates or engine == "ai") and ocr_engine.use_easyocr:
            try:
                ai_results = ocr_engine.reader.readtext(img)
                for bbox, text, conf in ai_results:
                    clean_t = "".join([c.upper() for c in text if c.isalnum()])
                    if len(clean_t) >= 4 and float(conf) >= 0.2:
                        pts = np.array(bbox, dtype=np.int32)
                        x_min, y_min = pts.min(axis=0)
                        x_max, y_max = pts.max(axis=0)
                        
                        # Add margin around box
                        h_box, w_box = y_max - y_min, x_max - x_min
                        x_min_pad = max(0, int(x_min - 0.1 * w_box))
                        y_min_pad = max(0, int(y_min - 0.1 * h_box))
                        x_max_pad = min(img.shape[1], int(x_max + 0.1 * w_box))
                        y_max_pad = min(img.shape[0], int(y_max + 0.1 * h_box))
                        
                        box_coords = [x_min_pad, y_min_pad, x_max_pad - x_min_pad, y_max_pad - y_min_pad]
                        crop_img = img[y_min_pad:y_max_pad, x_min_pad:x_max_pad]
                        
                        # Avoid duplicate box if already in raw_candidates
                        is_dup = False
                        for existing in raw_candidates:
                            ex_b = existing["box"]
                            if abs(ex_b[0] - box_coords[0]) < 50 and abs(ex_b[1] - box_coords[1]) < 50:
                                is_dup = True
                                break
                        if not is_dup and crop_img.size > 0:
                            cleaned_t = ocr_engine.clean_plate_text(text)
                            raw_candidates.append({
                                "text": cleaned_t if cleaned_t else clean_t,
                                "clean_text": clean_t,
                                "confidence": float(conf),
                                "box": box_coords,
                                "cropped_plate": crop_img
                            })
            except Exception as e:
                print(f"Whole-image EasyOCR fallback error: {e}")
        
        # Helper for scoring license plate candidates
        img_h, img_w = img.shape[:2]
        max_dist = np.sqrt(img_w**2 + img_h**2) / 2.0

        def score_plate(p):
            txt = p["clean_text"]
            c = p["confidence"]
            coords = p["box"]
            
            if len(txt) < 3:
                return 0.0
                
            # Compute distance from center of image
            cx, cy = coords[0] + coords[2]/2.0, coords[1] + coords[3]/2.0
            dist = np.sqrt((cx - img_w/2.0)**2 + (cy - img_h/2.0)**2)
            center_factor = max(0.4, 1.25 - 0.75 * (dist / max_dist))
            
            # Prefer standard plate lengths (5 to 8 characters)
            length_factor = 1.0 + 0.35 * min(len(txt), 8)
            
            return c * length_factor * center_factor

        # Sort raw_candidates by candidate quality score
        raw_candidates.sort(key=score_plate, reverse=True)
        
        # Add only valid top candidates to history database and response
        plates_found = []
        for p in raw_candidates:
            record = add_history_record(p["text"], p["cropped_plate"], p["confidence"])
            plates_found.append({
                "text": p["text"],
                "confidence": p["confidence"],
                "box": p["box"],
                "status": record["status"],
                "cropped_url": record["image_url"]
            })
        
        # Build Response
        response = {
            "success": len(plates_found) > 0,
            "plates": plates_found,
            "steps": cv_res["step_images"] # Contains all intermediate step base64 data URLs
        }
        
        # Add single character crops to steps for step 8 display
        if cv_res["plates"] and cv_res["characters"]:
            char_crops_b64 = [to_base64(c) for c in cv_res["characters"][0] if c is not None]
            response["segmented_characters"] = char_crops_b64
        else:
            response["segmented_characters"] = []

        return response

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Exception in process-image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-frame")
def process_frame(data: WebcamFrame):
    try:
        # Decode base64 image
        header, encoded = data.image.split(",", 1)
        contents = base64.b64decode(encoded)
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid base64 frame.")
            
        # Initialize Plate Finder
        finder = PlateFinder(
            min_plate_area=data.min_area,
            max_plate_area=data.max_area,
            min_ratio=data.min_ratio,
            max_ratio=data.max_ratio,
            blur_kernel=data.blur_kernel,
            sobel_kernel=data.sobel_kernel,
            morph_w=data.morph_w,
            morph_h=data.morph_h
        )
        
        # Run CV Pipeline (no need for all steps in real-time webcam to save bandwidth)
        cv_res = finder.process_frame(img)
        
        plates_found = []
        for i, cropped_plate in enumerate(cv_res["plates"]):
            coords = cv_res["coordinates"][i]
            
            if data.engine == "classic":
                segmented_chars = cv_res["characters"][i]
                plate_text, conf = ocr_engine.recognize_characters(segmented_chars)
                if not plate_text:
                    plate_text, conf = ocr_engine.recognize_plate(cropped_plate)
            else:
                plate_text, conf = ocr_engine.recognize_plate(cropped_plate)
                
            conf = float(conf)
            if plate_text.strip() and len(plate_text.replace(" ", "")) >= 4:
                record = add_history_record(plate_text, cropped_plate, conf)
                plates_found.append({
                    "text": plate_text,
                    "confidence": conf,
                    "box": coords,
                    "status": record["status"],
                    "cropped_url": record["image_url"]
                })
                
        return {
            "success": len(plates_found) > 0,
            "plates": plates_found
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Exception in process-frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-video")
async def upload_video(
    video: UploadFile = File(...),
    engine: str = Form("classic"),
    min_area: int = Form(1000),
    max_area: int = Form(30000),
    min_ratio: float = Form(2.5),
    max_ratio: float = Form(7.0),
    blur_kernel: int = Form(7),
    sobel_kernel: int = Form(3),
    morph_w: int = Form(22),
    morph_h: int = Form(3),
):
    try:
        # Save temp file
        temp_input_filename = f"temp_input_{uuid.uuid4().hex[:8]}.mp4"
        temp_input_path = os.path.join(VIDEOS_DIR, temp_input_filename)
        
        with open(temp_input_path, "wb") as f:
            f.write(await video.read())
            
        # Open video capture
        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Unable to read video file.")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        
        # Setup Video Writer (use MP4V codec which is standard and write as .mp4)
        output_filename = f"processed_{uuid.uuid4().hex[:8]}.mp4"
        output_path = os.path.join(VIDEOS_DIR, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        finder = PlateFinder(
            min_plate_area=min_area,
            max_plate_area=max_area,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            blur_kernel=blur_kernel,
            sobel_kernel=sobel_kernel,
            morph_w=morph_w,
            morph_h=morph_h
        )
        
        plates_detected_during_video = []
        frame_idx = 0
        
        # Processing loop (skip frames to speed up computation on CPU if desired)
        # We process every 2nd or 3rd frame for license plate but draw overlay on all
        # To make it fast, we can process every frame, but since it's a demo, we process every frame.
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            # Run finder
            # We run finder every 3 frames to avoid killing CPU in EasyOCR, but we draw overlays on all frames based on last detection.
            # To be simple and robust, let's process every 3 frames for OCR and draw the boxes.
            cv_res = finder.process_frame(frame)
            
            # Draw bounding boxes and text
            for i, cropped_plate in enumerate(cv_res["plates"]):
                coords = cv_res["coordinates"][i]
                x, y, w, h = coords
                
                # Draw box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 153, 255), 3)
                
                # Perform OCR
                if engine == "classic":
                    segmented_chars = cv_res["characters"][i]
                    plate_text, conf = ocr_engine.recognize_characters(segmented_chars)
                    if not plate_text:
                        plate_text, conf = ocr_engine.recognize_plate(cropped_plate)
                else:
                    plate_text, conf = ocr_engine.recognize_plate(cropped_plate)
                    
                conf = float(conf)
                if plate_text.strip():
                    plate_text_clean = plate_text.replace(" ", "").upper()
                    # Only add if it's not a duplicate from the last few frames
                    if not plates_detected_during_video or plates_detected_during_video[-1]["text"] != plate_text:
                        record = add_history_record(plate_text, cropped_plate, conf)
                        plates_detected_during_video.append({
                            "text": plate_text,
                            "confidence": conf,
                            "status": record["status"],
                            "timestamp": record["timestamp"]
                        })
                    
                    # Draw text label overlay
                    label = f"{plate_text} ({round(conf*100)}%)"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 153, 255), 2)
                    
            out.write(frame)
            
        cap.release()
        out.release()
        
        # Clean up input file
        try:
            os.remove(temp_input_path)
        except Exception:
            pass
            
        return {
            "success": True,
            "processed_video_url": f"/static/videos/{output_filename}",
            "plates_detected": plates_detected_during_video
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Exception in process-video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-sample")
def process_sample(
    engine: str = Form("classic"),
    min_area: int = Form(1000),
    max_area: int = Form(30000),
    min_ratio: float = Form(2.5),
    max_ratio: float = Form(7.0),
    blur_kernel: int = Form(7),
    sobel_kernel: int = Form(3),
    morph_w: int = Form(22),
    morph_h: int = Form(3),
):
    try:
        parent_dir = os.path.dirname(BASE_DIR)
        sample_path = os.path.join(parent_dir, "test_car.mp4")
        
        if not os.path.exists(sample_path):
            from generate_test_video import create_synthetic_video
            create_synthetic_video(sample_path)
            
        cap = cv2.VideoCapture(sample_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Unable to read sample video file.")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        
        output_filename = f"processed_sample_{uuid.uuid4().hex[:8]}.mp4"
        output_path = os.path.join(VIDEOS_DIR, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        finder = PlateFinder(
            min_plate_area=min_area,
            max_plate_area=max_area,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            blur_kernel=blur_kernel,
            sobel_kernel=sobel_kernel,
            morph_w=morph_w,
            morph_h=morph_h
        )
        
        plates_detected_during_video = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            cv_res = finder.process_frame(frame)
            
            for i, cropped_plate in enumerate(cv_res["plates"]):
                coords = cv_res["coordinates"][i]
                x, y, w, h = coords
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 153, 255), 3)
                
                if engine == "classic":
                    segmented_chars = cv_res["characters"][i]
                    plate_text, conf = ocr_engine.recognize_characters(segmented_chars)
                    if not plate_text:
                        plate_text, conf = ocr_engine.recognize_plate(cropped_plate)
                else:
                    plate_text, conf = ocr_engine.recognize_plate(cropped_plate)
                    
                conf = float(conf)
                if plate_text.strip():
                    if not plates_detected_during_video or plates_detected_during_video[-1]["text"] != plate_text:
                        record = add_history_record(plate_text, cropped_plate, conf)
                        plates_detected_during_video.append({
                            "text": plate_text,
                            "confidence": conf,
                            "status": record["status"],
                            "timestamp": record["timestamp"]
                        })
                    
                    label = f"{plate_text} ({round(conf*100)}%)"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 153, 255), 2)
                    
            out.write(frame)
            
        cap.release()
        out.release()
        
        return {
            "success": True,
            "processed_video_url": f"/static/videos/{output_filename}",
            "plates_detected": plates_detected_during_video
        }
    except Exception as e:
        print(f"Exception in process-sample: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Whitelist Endpoints
@app.get("/api/whitelist")
def get_whitelist():
    return whitelist

@app.post("/api/whitelist")
def add_to_whitelist(item: WhitelistItem):
    global whitelist
    clean_plate = item.plate.strip().upper()
    if clean_plate and clean_plate not in whitelist:
        whitelist.append(clean_plate)
        with open(WHITELIST_PATH, "w") as f:
            json.dump(whitelist, f)
            
        # Re-check statuses in history
        update_history_statuses()
        
    return whitelist

@app.delete("/api/whitelist/{plate}")
def remove_from_whitelist(plate: str):
    global whitelist
    clean_plate = plate.strip().upper()
    if clean_plate in whitelist:
        whitelist.remove(clean_plate)
        with open(WHITELIST_PATH, "w") as f:
            json.dump(whitelist, f)
            
        # Re-check statuses in history
        update_history_statuses()
        
    return whitelist

def update_history_statuses():
    global history
    whitelist_clean = [p.replace(" ", "").upper() for p in whitelist]
    for record in history:
        clean_rec = record["plate"].replace(" ", "").upper()
        record["status"] = "Authorized" if clean_rec in whitelist_clean else "Unauthorized"
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f)

# History Endpoints
@app.get("/api/history")
def get_history():
    return history

@app.post("/api/clear-history")
def clear_history():
    global history
    history = []
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f)
        
    # Clear physical cropped images to save disk space
    for file in os.listdir(PLATES_DIR):
        if file.endswith(".jpg"):
            try:
                os.remove(os.path.join(PLATES_DIR, file))
            except Exception:
                pass
                
    return history

# Serve Frontend SPA (must be mounted last)
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
