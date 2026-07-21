import cv2
import numpy as np
from plate_finder import PlateFinder

img = cv2.imread("static/debug_rear_plate.jpg")
if img is not None:
    finder = PlateFinder()
    res = finder.process_frame(img)
    print(f"debug_rear_plate.jpg: found {len(res['plates'])} plates")
    for idx, (p, coords) in enumerate(zip(res['plates'], res['coordinates'])):
        print(f"  Plate #{idx}: coords={coords}, shape={p.shape}")
