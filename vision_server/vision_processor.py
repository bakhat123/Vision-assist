import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
import easyocr

class VisionProcessor:
    @staticmethod
    def iou(boxA, boxB):
        """
        Compute Intersection over Union (IoU) between two bounding boxes.
        boxA, boxB: [x1, y1, x2, y2]
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return interArea / float(boxAArea + boxBArea - interArea)

    def __init__(self):
        print("Loading AI Models... (This may take a moment on first run)")
        
        # 1. YOLOv8 Nano (object detection)
        self.yolo = YOLO("yolov8n.pt")
        
        # 2. EasyOCR (text detection)
        self.ocr = easyocr.Reader(['en'])
        
        # 3. Haar Cascade Face Detector (OpenCV)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

# ----------------------------------------------------
# real widths in cm of objects for distance estimation
# ----------------------------------------------------
        # Distance estimation parameters
        self.KNOWN_WIDTH = {
         "pen": 1.0,
         "pencil": 1.0,
         "eraser": 3.0,
         "sharpener": 4.0,
         "ruler": 30.0,
         "notebook small": 15.0,
         "notebook large": 21.0,
         "binder": 25.0,
         "scissors": 15.0,
         "stapler": 12.0,
         "tape dispenser": 15.0,
         "marker": 2.0,
         "highlighter": 2.0,
         "calculator": 15.0,
         "mouse wireless": 6.0,
         "keyboard small": 40.0,
         "keyboard large": 50.0,
         "monitor small": 40.0,
         "monitor large": 60.0,
         "printer": 50.0,
         "scanner": 40.0,
         "router": 20.0,
         "modem": 15.0,
         "external hard drive": 12.0,
         "usb stick": 2.0,
         "smartwatch": 5.0,
         "headset": 18.0,
         "speaker small": 15.0,
         "speaker large": 30.0,
         "microphone": 5.0,
         "camera small": 12.0,
         "camera dslr": 15.0,
         "tripod small": 10.0,
         "tripod large": 15.0,
         "guitar acoustic": 35.0,
         "guitar electric": 32.0,
         "violin": 30.0,
         "drum small": 35.0,
         "drum large": 50.0,
         "keyboard piano": 120.0,
         "basketball": 24.0,
         "soccer ball": 22.0,
         "tennis ball": 6.5,
         "tennis racket": 27.0,
         "baseball bat": 7.5,
         "helmet bike": 25.0,
         "helmet motorbike": 30.0,
         "backpack small": 25.0,
         "backpack large": 35.0,
         "suitcase small": 40.0,
         "suitcase large": 60.0,
         "handbag": 25.0,
         "wallet": 10.0,
         "shoe adult": 12.0,
         "shoe child": 8.0,
         "boot": 15.0,
         "sock": 5.0,
         "hat": 20.0,
         "glasses": 14.0,
         "sunglasses": 14.0,
         "umbrella closed": 8.0,
         "umbrella open": 100.0,
         "chair office": 50.0,
         "chair dining": 45.0,
         "stool": 35.0,
         "table small": 60.0,
         "table large": 150.0,
         "coffee table": 100.0,
         "sofa 2-seater": 150.0,
         "sofa 3-seater": 200.0,
         "bed single": 100.0,
         "bed double": 140.0,
         "bed queen": 160.0,
         "bed king": 180.0,
         "pillow": 50.0,
         "blanket": 150.0,
         "curtain small": 80.0,
         "curtain large": 150.0,
         "door": 90.0,
         "window small": 60.0,
         "window large": 120.0,
         "refrigerator small": 60.0,
         "refrigerator large": 80.0,
         "microwave": 50.0,
         "oven": 60.0,
         "sink": 60.0,
         "washing machine": 70.0,
         "dishwasher": 60.0,
         "toaster": 25.0,
         "kettle": 20.0,
         "lamp desk": 20.0,
         "lamp floor": 40.0,
         "clock wall": 30.0,
         "clock desk": 15.0,
         "vase small": 10.0,
         "vase large": 20.0,
         "flower pot": 20.0,
         "bottle water": 7.0,
         "bottle soda": 7.0,
         "cup small": 8.0,
         "cup large": 10.0,
         "plate small": 20.0,
         "plate large": 25.0,
         "bowl": 20.0,
         "spoon": 5.0,
         "fork": 5.0,
         "knife": 5.0,
         "chair stool": 35.0,
         "table round": 120.0,
    }

# ----------------------------------------------------
# Definig focal length in pixels for distance estimation will change it later
# ----------------------------------------------------
        self.FOCAL_LENGTH_PX = 850.0

        print("Models loaded successfully.")

    def remove_overlapping_objects(self, objects):
        """
        Remove duplicate objects that overlap significantly (IoU > 0.5) and share the same label.
        """
        filtered_objects = []
        for obj in objects:
            keep = True
            for f in filtered_objects:
                if obj["label"] == f["label"] and self.iou(obj["bbox"], f["bbox"]) > 0.5:
                    keep = False
                    break
            if keep:
                filtered_objects.append(obj)
        return filtered_objects

    def process_pil(self, pil_img: Image.Image):
        """
        Process a PIL image: detect objects, texts, faces, and estimate distances.
        """
        # Convert PIL to OpenCV BGR
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        out = {"objects": [], "texts": [], "faces": [], "distances": {}}

# ----------------------------------------------------
        # --- 1) Object Detection ---
# ----------------------------------------------------

        dets = self.yolo(img, imgsz=640, verbose=False)[0]
        objects = []
        for box, cls, conf in zip(dets.boxes.xyxy, dets.boxes.cls, dets.boxes.conf):
            x1, y1, x2, y2 = map(int, box.tolist())
            label = self.yolo.model.names[int(cls)]
            objects.append({
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "score": round(float(conf), 2),
                "distance_cm": None
            })

        # Remove overlapping duplicates
        objects = self.remove_overlapping_objects(objects)
        out["objects"] = objects

        # --- 2) OCR (EasyOCR) ---
        ocr_results = self.ocr.readtext(np.array(pil_img), detail=0)  # returns only text
        out["texts"] = ocr_results

        # --- 3) Face Detection (Haar Cascade) ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        faces_list = [{"bbox": [int(x), int(y), int(x + w), int(y + h)], "status": "unknown"} 
                      for (x, y, w, h) in detected_faces]
        out["faces"] = faces_list

# ----------------------------------------------------
        # --- 4) Distance Estimation --- according to the formula
# ----------------------------------------------------
        height_px = img.shape[0]
        for o in out["objects"]:
            x1, y1, x2, y2 = o["bbox"]
            bbox_w = max(1, x2 - x1)
            bbox_h = max(1, y2 - y1)
            label = o["label"]
            distance_cm = None

            try:
                if label in self.KNOWN_WIDTH:
                    real_w_cm = float(self.KNOWN_WIDTH[label])
                    distance_cm = round((real_w_cm * self.FOCAL_LENGTH_PX) / float(bbox_w), 1)
                else:
                    # Fallback: simple height proxy
                    proxy = 1.0 / (bbox_h / height_px + 0.1) * 3.0
                    distance_cm = round(proxy * 100.0, 1)
            except:
                proxy = 1.0 / (bbox_h / height_px + 0.1) * 3.0
                distance_cm = round(proxy * 100.0, 1)

            o["distance_cm"] = distance_cm
            out["distances"][label] = round(distance_cm / 100.0, 2)

        return out
