import cv2
from ocr_engine import OCREngine
from plate_finder import PlateFinder

print("Initializing test verification script...")
ocr = OCREngine()
finder = PlateFinder()

img = cv2.imread("static/debug_original.jpg")
if img is not None:
    res = finder.process_frame(img)
    print(f"Contour candidates found: {len(res['plates'])}")
    for idx, (p, coords) in enumerate(zip(res['plates'], res['coordinates'])):
        txt, conf = ocr.recognize_plate(p)
        print(f"Candidate #{idx}: coords={coords}, text='{txt}', conf={conf:.2f}")

    if ocr.use_easyocr:
        full_res = ocr.reader.readtext(img)
        print("\nDirect EasyOCR Whole-Image Detections:")
        for bbox, text, conf in full_res:
            print(f"  Text: '{text}', Conf: {conf:.2f}, Box: {bbox}")
else:
    print("static/debug_original.jpg not found")
