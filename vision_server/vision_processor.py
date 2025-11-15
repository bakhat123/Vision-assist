import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
import easyocr
import math

class VisionProcessor:
    def __init__(self):
        print("Loading AI Models... (This may take a moment on first run)")
        
        # 1. Object Detection Model (YOLOv8 Nano - small and fast)
        self.yolo = YOLO("yolov8n.pt") 
        
        # 2. OCR Reader (English only for now, easyocr will download models)
        self.ocr = easyocr.Reader(['en'])
        
        # 3. Simple Face Detector (OpenCV Haar Cascade is simple and local)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
        print("Models loaded successfully.")

    def process_pil(self, pil_img: Image.Image):
        # Convert PIL image to OpenCV format (BGR) for processing
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Initialize the output structure
        out = {"objects": [], "texts": [], "faces": [], "distances": {}}

        # --- 1) Object Detection (YOLOv8) ---
        # Run inference once
        dets = self.yolo(img, imgsz=640, verbose=False)[0] 
        objects = []
        for box, cls, conf in zip(dets.boxes.xyxy, dets.boxes.cls, dets.boxes.conf):
            x1, y1, x2, y2 = map(int, box.tolist())
            label = self.yolo.model.names[int(cls)]
            objects.append({
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "score": round(float(conf), 2)
            })
        out["objects"] = objects

        # --- 2) OCR (EasyOCR) ---
        # Note: EasyOCR uses the slower NumPy array format
        ocr_results = self.ocr.readtext(np.array(pil_img), detail=0) # detail=0 returns only text
        out["texts"] = ocr_results

        # --- 3) Faces (Haar Cascade Detector) ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Naive face detection, returns (x, y, w, h)
        detected_faces = self.face_cascade.detectMultiScale(gray, 1.1, 4) 
        
        faces_list = []
        for (x, y, w, h) in detected_faces:
            faces_list.append({"bbox": [int(x), int(y), int(x + w), int(y + h)], "status": "unknown"})
        out["faces"] = faces_list

        # --- 4) Rough Distance Estimation (Bbox Height Heuristic) ---
        # This is a placeholder: we assume larger object on screen = closer
        height_px = img.shape[0]
        for o in objects:
            x1, y1, x2, y2 = o["bbox"]
            bbox_h = y2 - y1
            
            # Simple inverse relationship (larger height -> smaller normalized value)
            # This value is a proxy, not a real meter measurement! We refine in Week 2.
            normalized_depth_proxy = 1.0 / (bbox_h / height_px + 0.1) * 3.0
            out["distances"][o["label"]] = round(float(normalized_depth_proxy), 1)

        return out