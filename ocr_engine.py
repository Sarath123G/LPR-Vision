import sys
import numpy as np
import cv2
import traceback

_CACHED_TEMPLATES = None
_CACHED_EASYOCR_READER = None

class OCREngine:
    def __init__(self):
        global _CACHED_TEMPLATES, _CACHED_EASYOCR_READER
        self.use_easyocr = False
        self.reader = None
        
        # Ensure sys.stdout handles UTF-8 on Windows to prevent charmap errors during downloads
        try:
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

        if _CACHED_EASYOCR_READER is not None:
            self.reader = _CACHED_EASYOCR_READER
            self.use_easyocr = True
        else:
            try:
                import easyocr
                print("[OCR Engine] Initializing EasyOCR neural engine...")
                self.reader = easyocr.Reader(['en'], gpu=False)
                _CACHED_EASYOCR_READER = self.reader
                self.use_easyocr = True
                print("[OCR Engine] EasyOCR neural engine initialized successfully!")
            except Exception as e:
                print(f"[OCR Engine] EasyOCR initialization failed: {e}. Falling back to template matching.")
                self.use_easyocr = False

        if not self.use_easyocr or _CACHED_TEMPLATES is None:
            if _CACHED_TEMPLATES is None:
                _CACHED_TEMPLATES = self._generate_templates()
            self.templates = _CACHED_TEMPLATES
        else:
            self.templates = _CACHED_TEMPLATES

    def _generate_templates(self):
        """
        Generates multi-font, aspect-ratio preserved reference templates in memory
        for alphanumeric characters 0-9 and A-Z.
        """
        templates = {}
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        fonts = [
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_TRIPLEX,
            cv2.FONT_HERSHEY_COMPLEX
        ]
        thicknesses = [2, 3, 4, 5]
        
        for char in chars:
            char_tpls = []
            for font in fonts:
                for thick in thicknesses:
                    img = np.zeros((65, 65), dtype=np.uint8)
                    cv2.putText(img, char, (10, 50), font, 1.4, 255, thick, cv2.LINE_AA)
                    pts = np.argwhere(img > 0)
                    if pts.size > 0:
                        y_min, x_min = pts.min(axis=0)
                        y_max, x_max = pts.max(axis=0)
                        cropped = img[y_min:y_max+1, x_min:x_max+1]
                        h, w = cropped.shape
                        ar = w / float(h) if h > 0 else 0.5
                        
                        new_w = min(24, max(4, int(40 * ar)))
                        resized_c = cv2.resize(cropped, (new_w, 40))
                        padded = np.zeros((40, 24), dtype=np.uint8)
                        start_x = (24 - new_w) // 2
                        padded[:, start_x:start_x+new_w] = resized_c
                        char_tpls.append((padded, padded > 128))
                    else:
                        r = cv2.resize(img, (24, 40))
                        char_tpls.append((r, r > 128))
            templates[char] = char_tpls
            
        return templates

    def clean_plate_text(self, text):
        """
        Cleans and post-corrects license plate OCR text using structural context and rules.
        """
        if not text:
            return ""
            
        raw = "".join([c.upper() for c in text if c.isalnum()])
        if not raw:
            return ""
            
        to_digit = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6'}
        to_letter = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G'}
        
        chars = list(raw)
        n = len(chars)
        
        # Rule A: Indian / Global format 7-10 char plates (e.g. DL3CAG1234)
        if n >= 7:
            # Last 4 positions are digits (e.g. 1234)
            for idx in range(max(0, n - 4), n):
                if chars[idx] in to_digit:
                    chars[idx] = to_digit[chars[idx]]
                    
        # Rule B: Digit flanked by digits
        for i in range(1, n - 1):
            if chars[i-1].isdigit() and chars[i+1].isdigit():
                if chars[i] in to_digit:
                    chars[i] = to_digit[chars[i]]
                    
        return "".join(chars)

    def _match_single_character(self, char_img):
        """
        Performs template matching using combined CCOEFF + IoU metrics against multi-font templates.
        """
        if char_img is None or char_img.size == 0:
            return "?", 0.0
            
        try:
            # Convert to grayscale
            if len(char_img.shape) == 3:
                gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = char_img.copy()
                
            # Binarize character image using Otsu thresholding
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Ensure background is black and foreground text is white
            h, w = thresh.shape
            corners = [int(thresh[0,0]), int(thresh[0, w-1]), int(thresh[h-1, 0]), int(thresh[h-1, w-1])]
            if sum(corners) >= 510 or np.mean(thresh) > 120:
                thresh = cv2.bitwise_not(thresh)
                
            # Crop using largest contour
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                largest = max(cnts, key=cv2.contourArea)
                cx, cy, cw, ch_w = cv2.boundingRect(largest)
                cropped = thresh[cy:cy+ch_w, cx:cx+cw]
                if cropped.shape[0] > 2 and cropped.shape[1] > 2:
                    h_c, w_c = cropped.shape
                    ar = w_c / float(h_c) if h_c > 0 else 0.5
                    new_w = min(24, max(4, int(40 * ar)))
                    resized_c = cv2.resize(cropped, (new_w, 40))
                    resized = np.zeros((40, 24), dtype=np.uint8)
                    start_x = (24 - new_w) // 2
                    resized[:, start_x:start_x+new_w] = resized_c
                else:
                    resized = cv2.resize(thresh, (24, 40))
            else:
                resized = cv2.resize(thresh, (24, 40))
                
            bin_img = (resized > 128)
            best_char = "?"
            best_combo = -1.0
            
            # Find closest match across template variants using combined metric
            for char, tpl_list in self.templates.items():
                for tpl_img, tpl_bin in tpl_list:
                    res = cv2.matchTemplate(resized, tpl_img, cv2.TM_CCOEFF_NORMED)
                    cc_score = max(0.0, float(res[0][0]))
                    
                    intersection = np.logical_and(bin_img, tpl_bin).sum()
                    union = np.logical_or(bin_img, tpl_bin).sum()
                    iou_score = intersection / float(union) if union > 0 else 0.0
                    
                    combo = 0.4 * cc_score + 0.6 * iou_score
                    if combo > best_combo:
                        best_combo = combo
                        best_char = char
                        
            # Normalize confidence score to realistic 0.0 - 0.98 scale
            confidence = min(0.98, max(0.0, best_combo * 1.4))
            return best_char, confidence
            
        except Exception as e:
            print(f"Error matching template: {e}")
            return "?", 0.0

    def recognize_plate(self, plate_img, use_ai=True):
        """
        Performs LPR on whole plate image using neural EasyOCR if available,
        falling back to traditional character segmentation + template matching.
        """
        if plate_img is None or plate_img.size == 0:
            return "", 0.0
            
        if self.use_easyocr:
            try:
                # Add white padding around crop & upscale if height is low to boost OCR accuracy
                h, w = plate_img.shape[:2]
                target_img = plate_img
                if h < 60:
                    scale = 60.0 / h
                    target_img = cv2.resize(plate_img, (int(w * scale), 60), interpolation=cv2.INTER_CUBIC)
                
                h_t, w_t = target_img.shape[:2]
                padded_img = cv2.copyMakeBorder(target_img, int(h_t*0.2), int(h_t*0.2), int(w_t*0.2), int(w_t*0.2), cv2.BORDER_CONSTANT, value=[255,255,255])
                
                results = self.reader.readtext(
                    padded_img, 
                    allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ- /"
                )
                if results:
                    texts = []
                    confidences = []
                    for bbox, text, conf in results:
                        clean_text = "".join([c.upper() for c in text if (c.isalnum() or c in ['-', '/'])])
                        if clean_text:
                            texts.append(clean_text)
                            confidences.append(float(conf))
                            
                    if texts:
                        combined_text = " ".join(texts)
                        avg_conf = float(np.mean(confidences))
                        cleaned_final = self.clean_plate_text(combined_text)
                        return (cleaned_final if cleaned_final else combined_text), avg_conf
            except Exception as e:
                print(f"EasyOCR recognize_plate failed: {e}")
                
        # TRADITIONAL CV TEMPLATE MATCHING SEGMENTATION
        try:
            if len(plate_img.shape) == 3:
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_img.copy()
                
            h_c, w_c = gray.shape[:2]
            if h_c < 10 or w_c < 15:
                return "", 0.0
                
            # Trim 8% top & bottom frame borders
            gray_trimmed = gray[int(h_c*0.08):int(h_c*0.92), int(w_c*0.03):int(w_c*0.97)]
            plate_trimmed = plate_img[int(h_c*0.08):int(h_c*0.92), int(w_c*0.03):int(w_c*0.97)]
            
            scale = 80.0 / gray_trimmed.shape[0]
            canonical_w = int(gray_trimmed.shape[1] * scale)
            gray_resized = cv2.resize(gray_trimmed, (canonical_w, 80))
            plate_resized = cv2.resize(plate_trimmed, (canonical_w, 80))
            
            # CLAHE Contrast Enhancement
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            gray_clahe = clahe.apply(gray_resized)
            
            # Binarize with Otsu INV
            _, thresh_inv = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Sobel X for vertical character edge detection
            sobelx = cv2.Sobel(gray_clahe, cv2.CV_8U, 1, 0, ksize=3)
            _, thresh_s = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Fuse character strokes horizontally
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
            thresh_fused = cv2.morphologyEx(thresh_s, cv2.MORPH_CLOSE, kernel_h)
            
            col_sums = np.sum(thresh_fused > 0, axis=0)
            in_char = False
            start_x = 0
            intervals = []
            min_pixels = max(1, int(0.05 * 80))
            
            for cx, val in enumerate(col_sums):
                if val >= min_pixels and not in_char:
                    in_char = True
                    start_x = cx
                elif val < min_pixels and in_char:
                    in_char = False
                    if cx - start_x >= 4:
                        intervals.append((start_x, cx))
            if in_char and (len(col_sums) - start_x >= 4):
                intervals.append((start_x, len(col_sums)))
                
            merged = []
            for sx, ex in intervals:
                if not merged:
                    merged.append((sx, ex))
                else:
                    prev_s, prev_e = merged[-1]
                    gap = sx - prev_e
                    prev_w = prev_e - prev_s
                    if gap < 5 or (prev_w < 12 and gap < 8):
                        merged[-1] = (prev_s, ex)
                    else:
                        merged.append((sx, ex))
                        
            final_intervals = [b for b in merged if (b[1] - b[0]) >= 6]
            
            plate_text = ""
            confs = []
            for sx, ex in final_intervals:
                c_bin = thresh_inv[:, max(0, sx-1):min(thresh_inv.shape[1], ex+1)]
                cnts, _ = cv2.findContours(c_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    max_c = max(cnts, key=cv2.contourArea)
                    bx, by, bw, bh = cv2.boundingRect(max_c)
                    if bh >= 16:
                        char_stroke = c_bin[by:by+bh, bx:bx+bw]
                        h_s, w_s = char_stroke.shape
                        ar_crop = w_s / float(h_s) if h_s > 0 else 0.5
                        
                        new_w = min(24, max(4, int(40 * ar_crop)))
                        resized_c = cv2.resize(char_stroke, (new_w, 40))
                        padded = np.zeros((40, 24), dtype=np.uint8)
                        start_x_p = (24 - new_w) // 2
                        padded[:, start_x_p:start_x_p+new_w] = resized_c
                        bin_img = (padded > 128)
                        
                        best_char = "?"
                        best_score = -1.0
                        for char, tpl_list in self.templates.items():
                            if bw >= 15 and char in ['I', '1']:
                                continue
                            for tpl_img, tpl_bin in tpl_list:
                                res = cv2.matchTemplate(padded, tpl_img, cv2.TM_CCOEFF_NORMED)
                                cc = max(0.0, float(res[0][0]))
                                intersection = np.logical_and(bin_img, tpl_bin).sum()
                                union = np.logical_or(bin_img, tpl_bin).sum()
                                iou = intersection / float(union) if union > 0 else 0.0
                                
                                score = 0.5 * cc + 0.5 * iou
                                if score > best_score:
                                    best_score = score
                                    best_char = char
                        if best_char != "?":
                            plate_text += best_char
                            confs.append(best_score)
                            
            avg_conf = float(np.mean(confs)) if confs else 0.0
            cleaned_text = self.clean_plate_text(plate_text)
            return cleaned_text, avg_conf
            
        except Exception as e:
            print(f"Fallback recognize_plate error: {e}")
            return "", 0.0

    def recognize_characters(self, char_imgs):
        """
        Recognize segmented character crops list.
        """
        if not char_imgs:
            return "", 0.0
            
        if self.use_easyocr:
            try:
                plate_text = ""
                total_conf = 0.0
                valid_chars = 0
                for img in char_imgs:
                    if img is None or img.size == 0:
                        continue
                    h, w = img.shape[:2]
                    padded = cv2.copyMakeBorder(img, int(h*0.2), int(h*0.2), int(w*0.2), int(w*0.2), cv2.BORDER_CONSTANT, value=[255,255,255])
                    results = self.reader.readtext(padded, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                    if results:
                        best = max(results, key=lambda x: x[2])
                        alphanum = [c for c in best[1].upper() if c.isalnum()]
                        if alphanum:
                            plate_text += alphanum[0]
                            total_conf += best[2]
                            valid_chars += 1
                return plate_text, (total_conf / valid_chars if valid_chars > 0 else 0.0)
            except Exception as e:
                print(f"EasyOCR recognize_characters error: {e}")
                
        # FALLBACK TEMPLATE MATCHING
        try:
            plate_text = ""
            total_conf = 0.0
            count = 0
            
            for img in char_imgs:
                char_str, conf = self._match_single_character(img)
                if char_str != "?":
                    plate_text += char_str
                    total_conf += conf
                    count += 1
                    
            avg_conf = total_conf / count if count > 0 else 0.0
            cleaned_text = self.clean_plate_text(plate_text)
            return cleaned_text, avg_conf
        except Exception as e:
            print(f"Fallback recognize_characters error: {e}")
            return "", 0.0
