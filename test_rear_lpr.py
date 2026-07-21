import cv2
import numpy as np
import os
from plate_finder import PlateFinder
from ocr_engine import OCREngine

def main():
    img_path = "static/debug_original.jpg"
    img = cv2.imread(img_path)
    
    # Standard settings from the screenshot
    finder = PlateFinder(
        min_plate_area=500,
        max_plate_area=30000,
        min_ratio=2.5,
        max_ratio=7.0,
        blur_kernel=7,
        sobel_kernel=3,
        morph_w=22,
        morph_h=3
    )
    
    # Run preprocessing to analyze
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (finder.blur_kernel, finder.blur_kernel), 0)
    sobel = cv2.Sobel(blurred, cv2.CV_8U, 1, 0, ksize=finder.sobel_kernel)
    _, thresholded = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (finder.morph_w, finder.morph_h))
    closed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, morph_kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(f"Found {len(contours)} raw contours.")
    
    ocr = OCREngine()
    for idx, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area > 100:
            plate, chars, coords, char_mask = finder.check_plate(img, cnt)
            if plate is not None:
                text, conf = ocr.recognize_plate(plate)
                print(f"Detected plate at {coords}: text='{text}' with conf {conf:.2f}")
            else:
                pass

if __name__ == "__main__":
    main()
