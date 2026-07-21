import cv2
import numpy as np
import base64
from skimage import measure
import imutils

def to_base64(img):
    if img is None or img.size == 0:
        return ""
    try:
        _, buffer = cv2.imencode('.jpg', img)
        return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error in base64 encoding: {e}")
        return ""

def sort_cont(character_contours):
    """Sort contours left-to-right based on bounding box x-coordinate"""
    if not character_contours:
        return []
    boundingBoxes = [cv2.boundingRect(c) for c in character_contours]
    sorted_pairs = sorted(zip(character_contours, boundingBoxes), key=lambda b: b[1][0])
    character_contours, _ = zip(*sorted_pairs)
    return list(character_contours)

def segment_chars(plate_img, fixed_width=400):
    """
    Extract Value channel from HSV format of image and apply adaptive thresholding
    to reveal and segment characters on the license plate.
    """
    if plate_img is None or plate_img.size == 0:
        return None, None

    # Step 1: Convert to HSV and extract V channel
    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
    V = cv2.split(hsv)[2]

    # Step 2: Adaptive Thresholding & Bitwise Not
    thresh = cv2.adaptiveThreshold(V, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    # Resize for canonical processing
    plate_img_resized = imutils.resize(plate_img, width=fixed_width)
    thresh_resized = imutils.resize(thresh, width=fixed_width)
    bgr_thresh = cv2.cvtColor(thresh_resized, cv2.COLOR_GRAY2BGR)

    # Step 3: Connected component analysis & Character bounding boxes
    charCandidates = np.zeros(thresh_resized.shape, dtype='uint8')
    characters = []
    cnts, _ = cv2.findContours(thresh_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
        if boxH == 0 or boxW == 0:
            continue
            
        aspectRatio = boxW / float(boxH)
        solidity = cv2.contourArea(c) / float(boxW * boxH)
        heightRatio = boxH / float(plate_img_resized.shape[0])

        # Validation rules for character shapes
        keepAspectRatio = aspectRatio < 1.4
        keepSolidity = solidity > 0.10
        keepHeight = 0.25 < heightRatio < 0.95

        if keepAspectRatio and keepSolidity and keepHeight and boxW >= 4:
            hull = cv2.convexHull(c)
            cv2.drawContours(charCandidates, [hull], -1, 255, -1)

    # Find contours on clean mask containing character candidates
    contours, _ = cv2.findContours(charCandidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        contours = sort_cont(contours)
        addPixel = 4
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            
            # Pad the bounding box
            y_start = max(y - addPixel, 0)
            x_start = max(x - addPixel, 0)
            y_end = min(y + h + addPixel * 2, thresh_resized.shape[0])
            x_end = min(x + w + addPixel * 2, thresh_resized.shape[1])
            
            temp = plate_img_resized[y_start:y_end, x_start:x_end]
            if temp.size > 0:
                characters.append(temp)
        return characters, charCandidates
    
    return None, None

class PlateFinder:
    def __init__(self, 
                 min_plate_area=300, 
                 max_plate_area=50000, 
                 min_ratio=1.2, 
                 max_ratio=8.0,
                 blur_kernel=7,
                 sobel_kernel=3,
                 morph_w=22,
                 morph_h=3):
        self.min_area = min_plate_area
        self.max_area = max_plate_area
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.blur_kernel = blur_kernel
        self.sobel_kernel = sobel_kernel
        self.morph_w = morph_w
        self.morph_h = morph_h
        
        # Structure element for morphological operations
        self.element_structure = cv2.getStructuringElement(
            shape=cv2.MORPH_RECT, 
            ksize=(self.morph_w, self.morph_h)
        )

    def preprocess(self, input_img):
        # 1. Blur
        k = self.blur_kernel
        if k % 2 == 0:
            k += 1 # Ensure odd
        imgBlurred = cv2.GaussianBlur(input_img, (k, k), 0)
        
        # 2. Grayscale
        gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)
        
        # 3. Sobel Filter (vertical edges)
        sk = self.sobel_kernel
        if sk % 2 == 0:
            sk += 1
        sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=sk)
        
        # 4. Otsu Thresholding
        _, threshold_img = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 5. Morphological Closing
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_w, self.morph_h))
        morph_n_thresholded_img = cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE, kernel=element)
        
        return gray, sobelx, threshold_img, morph_n_thresholded_img

    def clean_plate(self, plate):
        if plate is None or plate.size == 0:
            return plate, False, None
        
        h, w = plate.shape[:2]
        plate_area = w * h
        if not self.ratioCheck(plate_area, w, h):
            return plate, False, None
            
        return plate, True, [0, 0, w, h]

    def ratioCheck(self, area, width, height):
        ratio = float(width) / float(height) if height != 0 else 0
        if ratio < 1:
            ratio = 1 / ratio if ratio != 0 else 0
            
        area_ok = self.min_area <= area <= self.max_area
        ratio_ok = self.min_ratio <= ratio <= self.max_ratio
        
        if not area_ok or not ratio_ok:
            print(f"[DEBUG] ratioCheck candidate (area={area}, ratio={ratio:.2f}) rejected. Limits: area={self.min_area}-{self.max_area}, ratio={self.min_ratio:.2f}-{self.max_ratio:.2f}")
            
        return area_ok and ratio_ok

    def validateRatio(self, rect):
        (x, y), (width, height), rect_angle = rect
        if width < height:
            width, height = height, width
            
        if height == 0 or width == 0:
            return False
            
        area = width * height
        ratio = float(width) / float(height)
            
        pre_ratio_ok = (self.min_ratio - 0.5) <= ratio <= (self.max_ratio + 1.5)
        area_ok = (self.min_area - 300) <= area <= (self.max_area + 30000)
        
        return pre_ratio_ok and area_ok

    def check_plate(self, input_img, contour):
        min_rect = cv2.minAreaRect(contour)
        if self.validateRatio(min_rect):
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            ratio = float(w) / float(h) if h > 0 else 0
            if ratio < 1.0:
                ratio = 1.0 / ratio if ratio > 0 else 0
                
            if area < self.min_area or area > (self.max_area + 25000) or ratio < (self.min_ratio - 0.4) or ratio > (self.max_ratio + 1.5):
                return None, None, None, None
                
            after_validation_img = input_img[y:y + h, x:x + w]
            
            after_clean_plate_img, plateFound, coordinates = self.clean_plate(after_validation_img)
            if plateFound and coordinates[2] > 0.4 * w and coordinates[3] > 0.4 * h:
                x1, y1, w1, h1 = coordinates
                final_crop = after_validation_img[y1:y1 + h1, x1:x1 + w1]
                abs_coords = [x1 + x, y1 + y, w1, h1]
            else:
                final_crop = after_validation_img
                abs_coords = [x, y, w, h]
                
            characters_on_plate, char_mask = segment_chars(final_crop)
            if characters_on_plate is None:
                characters_on_plate = []
            if char_mask is None:
                char_mask = np.zeros(final_crop.shape[:2], dtype=np.uint8)
                
            return final_crop, characters_on_plate, abs_coords, char_mask
        return None, None, None, None

    def process_frame(self, input_img):
        """
        Processes a single frame and returns a dict with:
        - plates: list of cropped plate images
        - coordinates: list of bounding boxes [x, y, w, h] on original image
        - characters: list of list of segmented character images
        - step_images: dict of base64 encoded images of intermediate steps
        """
        result = {
            "plates": [],
            "coordinates": [],
            "characters": [],
            "char_masks": [],
            "step_images": {}
        }
        
        # 1. Pipeline steps
        gray, sobelx, threshold_img, morph_n_thresholded_img = self.preprocess(input_img)
        
        # 2. Extract contours
        contours, _ = cv2.findContours(morph_n_thresholded_img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        
        # Save step images (resized to width 800 for fast base64 transfer)
        def fast_step_b64(img_step):
            if img_step is None or img_step.size == 0:
                return ""
            if img_step.shape[1] > 800:
                sc = 800.0 / img_step.shape[1]
                small = cv2.resize(img_step, (800, int(img_step.shape[0] * sc)))
                return to_base64(small)
            return to_base64(img_step)

        result["step_images"]["1_original"] = fast_step_b64(input_img)
        result["step_images"]["2_grayscale"] = fast_step_b64(gray)
        result["step_images"]["3_sobelx"] = fast_step_b64(sobelx)
        result["step_images"]["4_threshold"] = fast_step_b64(threshold_img)
        result["step_images"]["5_morphology"] = fast_step_b64(morph_n_thresholded_img)

        contour_img = input_img.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 153, 255), 2)
        result["step_images"]["6_contours"] = fast_step_b64(contour_img)

        # 3. Find possible plates
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if (self.min_area - 300) <= area <= (self.max_area + 30000):
                plate, chars, coords, char_mask = self.check_plate(input_img, cnt)
                if plate is not None:
                    result["plates"].append(plate)
                    result["coordinates"].append(coords)
                    result["characters"].append(chars)
                    result["char_masks"].append(char_mask)

        # Render step 7 and 8 visualizer output
        if result["plates"]:
            result["step_images"]["7_clean_plate"] = to_base64(result["plates"][0])
            result["step_images"]["8_character_mask"] = to_base64(result["char_masks"][0])
        else:
            result["step_images"]["7_clean_plate"] = ""
            result["step_images"]["8_character_mask"] = ""

        return result
