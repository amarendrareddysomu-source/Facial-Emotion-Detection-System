from ultralytics import YOLO
import cv2
import numpy as np

def remove_close_boxes(boxes, scores, min_dist=80):
    keep = []
    used = set()
    for i in range(len(boxes)):
        if i in used:
            continue
        for j in range(i+1, len(boxes)):
            # Box format [x, y, width, height]
            cx1 = boxes[i][0] + boxes[i][2] / 2.0
            cy1 = boxes[i][1] + boxes[i][3] / 2.0
            cx2 = boxes[j][0] + boxes[j][2] / 2.0
            cy2 = boxes[j][1] + boxes[j][3] / 2.0
            dist = ((cx1-cx2)**2 + (cy1-cy2)**2) ** 0.5
            if dist < min_dist:
                used.add(j if scores[i] >= scores[j] else i)
        if i not in used:
            keep.append(i)
    return keep

class HumanDetector:
    def __init__(self, model_path='yolov8s.pt'):
        try:
            # We use the small model ('yolov8s.pt') for better accuracy on close-ups
            self.model = YOLO(model_path)
            
            # COCO Animal Classes
            self.animal_classes = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            self.model = None

    def detect(self, frame):
        if self.model is None:
            return [], []
            
        # Get raw low confidence boxes to filter manually per class
        results = self.model(frame, verbose=False, conf=0.1, iou=0.99, agnostic_nms=False)
        frame_height, frame_width = frame.shape[:2]
        
        person_raw = []
        animal_raw = []
        object_raw = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                class_name = self.model.names[cls].capitalize()
                
                # Person Class
                if cls == 0:
                    if conf >= 0.55:
                        person_raw.append(([x1, y1, x2-x1, y2-y1], conf, 'person'))
                # Animal Class
                elif cls in self.animal_classes:
                    if conf >= 0.50:
                        # Dog box sanity check
                        if (x2 - x1) > 0.35 * frame_width or (y2 - y1) > 0.35 * frame_height:
                            continue
                        animal_raw.append(([x1, y1, x2-x1, y2-y1], conf, class_name))
                # Object Class
                else:
                    if conf >= 0.40:
                        object_raw.append(([x1, y1, x2-x1, y2-y1], conf, f'Other ({class_name})'))
                        
        person_detections = []
        ignored_objects = []
        
        # 1) Persons NMS (iou=0.30)
        if len(person_raw) > 0:
            boxes = [p[0] for p in person_raw]
            scores = [p[1] for p in person_raw]
            indices = cv2.dnn.NMSBoxes(boxes, scores, 0.55, 0.30)
            if len(indices) > 0:
                post_nms_raw = [person_raw[i] for i in indices.flatten()]
                post_nms_boxes = [p[0] for p in post_nms_raw]
                post_nms_scores = [p[1] for p in post_nms_raw]
                
                # Second-pass distance filter (80px radius deduplication cutoff)
                keep_indices = remove_close_boxes(post_nms_boxes, post_nms_scores, min_dist=80)
                for ki in keep_indices:
                    person_detections.append(post_nms_raw[ki])
                    
        # 2) Animals NMS (iou=0.50)
        if len(animal_raw) > 0:
            boxes = [a[0] for a in animal_raw]
            scores = [a[1] for a in animal_raw]
            indices = cv2.dnn.NMSBoxes(boxes, scores, 0.50, 0.50)
            if len(indices) > 0:
                for i in indices.flatten():
                    b = animal_raw[i][0]
                    ignored_objects.append(([b[0], b[1], b[0]+b[2], b[1]+b[3]], animal_raw[i][1], animal_raw[i][2]))
                    
        # 3) Objects NMS (iou=0.60)
        if len(object_raw) > 0:
            boxes = [o[0] for o in object_raw]
            scores = [o[1] for o in object_raw]
            indices = cv2.dnn.NMSBoxes(boxes, scores, 0.40, 0.60)
            if len(indices) > 0:
                for i in indices.flatten():
                    b = object_raw[i][0]
                    ignored_objects.append(([b[0], b[1], b[0]+b[2], b[1]+b[3]], object_raw[i][1], object_raw[i][2]))
                    
        return person_detections, ignored_objects
