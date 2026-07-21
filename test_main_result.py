import cv2
import numpy as np
from plate_finder import PlateFinder
from ocr_engine import OCREngine

def main():
    img = cv2.imread("static/debug_original.jpg")
    
    # Parameters from default uvicorn run
    engine = "classic"
    min_area = 500
    max_area = 30000
    min_ratio = 2.5
    max_ratio = 7.0
    blur_kernel = 7
    sobel_kernel = 3
    morph_w = 22
    morph_h = 3
    
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
    
    cv_res = finder.process_frame(img)
    print(f"Number of plates found: {len(cv_res['plates'])}")
    
    ocr_engine = OCREngine()
    plates_found = []
    for i, cropped_plate in enumerate(cv_res["plates"]):
        coords = cv_res["coordinates"][i]
        
        # classic ocr
        segmented_chars = cv_res["characters"][i]
        plate_text, conf = ocr_engine.recognize_characters(segmented_chars)
        
        whole_text, whole_conf = ocr_engine.recognize_plate(cropped_plate)
        if not plate_text or len(whole_text) > len(plate_text):
            plate_text, conf = whole_text, whole_conf
            
        if plate_text:
            plates_found.append({
                "text": plate_text,
                "confidence": float(conf),
                "box": coords
            })
            
    print("Raw plates found:")
    for p in plates_found:
        print(p)
        
    plates_found.sort(key=lambda x: (len(x["text"]), x["confidence"]), reverse=True)
    print("\nSorted plates found:")
    for p in plates_found:
        print(p)

if __name__ == "__main__":
    main()
