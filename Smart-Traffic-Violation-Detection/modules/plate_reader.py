import cv2
import numpy as np
import re

try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False)
    USE_EASYOCR = True
    print("[PlateReader] EasyOCR loaded successfully.")
except Exception:
    USE_EASYOCR = False
    print("[PlateReader] EasyOCR not available — using demo mode.")

# Indian number plate pattern
PLATE_PATTERN = re.compile(r'[A-Z]{2}\s?\d{2}\s?[A-Z]{1,2}\s?\d{4}')


class PlateReader:
    def __init__(self):
        self.last_plate = None

    def preprocess(self, roi):
        """Preprocess ROI for better OCR accuracy."""
        gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh  = cv2.threshold(blurred, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return thresh

    def extract_plate_region(self, frame):
        """Extract likely number plate region using contour detection."""
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)
        edges   = cv2.Canny(blurred, 30, 200)
        contours, _ = cv2.findContours(edges.copy(),
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        for c in contours:
            peri   = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect = w / float(h)
                if 2.0 < aspect < 6.0 and w > 60:
                    return frame[y:y+h, x:x+w], (x, y, w, h)
        return None, None

    def read(self, frame):
        """Read number plate text from frame."""
        if not USE_EASYOCR:
            # Demo: return simulated plate
            import random
            states = ['TN', 'MH', 'KA', 'DL', 'AP']
            state  = random.choice(states)
            return f"{state} {random.randint(1,99):02d} {random.choice('ABCDEFGH')}{random.choice('ABCDEFGH')} {random.randint(1000,9999)}"

        roi, bbox = self.extract_plate_region(frame)
        if roi is None:
            return self.last_plate

        try:
            processed = self.preprocess(roi)
            results   = reader.readtext(processed)
            for _, text, conf in results:
                text = text.upper().strip().replace(' ', '')
                # Try to match Indian plate pattern
                match = PLATE_PATTERN.search(text)
                if match:
                    plate_text = match.group()
                    self.last_plate = plate_text
                    return plate_text
                if conf > 0.4 and len(text) >= 6:
                    self.last_plate = text
                    return text
        except Exception as e:
            print(f"[PlateReader] OCR error: {e}")

        return self.last_plate
