import logging
logging.basicConfig(level=logging.DEBUG)

from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import pytesseract
import io
import base64
import time

app = Flask(__name__)

model = YOLO("assets/anpr2_yolov9_int8.tflite", task='detect')

def preprocess_plate_image(cropped_img):
    height, width = cropped_img.shape[:2]
    if height < 200:
        scale = 200 / height
        cropped_img = cv2.resize(cropped_img, (int(width * scale), 200), 
                               interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    _, binary = cv2.threshold(denoised, 0, 255, 
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return processed

def extract_plate_text(cropped_img):
    processed_img = preprocess_plate_image(cropped_img)
    
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    try:
        text = pytesseract.image_to_string(
            processed_img, 
            config=custom_config,
            timeout=2  # Prevent hangs on difficult images
        )
        return text.strip()
    except RuntimeError as e:
        logging.error(f"Tesseract error: {str(e)}")
        return ""

@app.route('/ocr', methods=['POST'])
def detect_from_base64():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "Image data missing"}), 400

    try:
        start_time = time.time()
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)

        results = model(image_np, conf=0.7)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return jsonify({"plate": "NOT_FOUND", "state": None})

        sorted_boxes = sorted(boxes, key=lambda b: b.conf.item(), reverse=True)
        
        for box in sorted_boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image_np.shape[1], x2 + padding)
            y2 = min(image_np.shape[0], y2 + padding)
            
            cropped = image_np[y1:y2, x1:x2]
            plate_text = extract_plate_text(cropped)

            if plate_text:
                plate_cleaned = ''.join(e for e in plate_text if e.isalnum())
                if len(plate_cleaned) >= 2:
                    state_code = plate_cleaned[:2]
                else:
                    state_code = None
                
                logging.info(f"Processing time: {time.time() - start_time:.2f}s")
                return jsonify({
                    "plate": plate_cleaned, 
                    "state": state_code,
                    "confidence": round(box.conf.item(), 2)
                })

        return jsonify({"plate": "NOT_FOUND", "state": None})

    except Exception as e:
        logging.exception("Processing error")
        return jsonify({"error": str(e)}), 500

@app.route('/ocr-file', methods=['POST'])
def detect_from_file_upload():
    start_time = time.time()
    logging.info("Received /ocr-file request")

    if 'image' not in request.files:
        return jsonify({"error": "Image file missing"}), 400

    try:
        file = request.files['image']
        image = Image.open(file.stream).convert("RGB")
        image_np = np.array(image)

        results = model(image_np, conf=0.7)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return jsonify({"plate": "NOT_FOUND", "state": None})

        sorted_boxes = sorted(boxes, key=lambda b: b.conf.item(), reverse=True)
        
        for box in sorted_boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image_np.shape[1], x2 + padding)
            y2 = min(image_np.shape[0], y2 + padding)
            
            cropped = image_np[y1:y2, x1:x2]
            plate_text = extract_plate_text(cropped)

            if plate_text:
                plate_cleaned = ''.join(e for e in plate_text if e.isalnum())
                state_code = plate_cleaned[:2] if len(plate_cleaned) >= 2 else None
                logging.info(f"Processing time: {time.time() - start_time:.2f}s")
                return jsonify({
                    "plate": plate_cleaned, 
                    "state": state_code,
                    "confidence": round(box.conf.item(), 2)
                })

        return jsonify({"plate": "NOT_FOUND", "state": None})

    except Exception as e:
        logging.exception(f"Processing error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)