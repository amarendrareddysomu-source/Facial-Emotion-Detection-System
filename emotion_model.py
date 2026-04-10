import os
import cv2
# Suppress TensorFlow logs which can be extremely verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
from deepface import DeepFace

class EmotionDetector:
    def __init__(self):
        print("EmotionDetector initialized.")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_emotion(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within frame boundaries
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        person_crop = frame[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return "Unknown", 0.0
            
        # CORRECT — detect face inside person box first, then predict emotion:
        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        
        if len(faces) > 0:
            fx, fy, fw, fh = faces[0]  # take the largest/most confident face
            
            # Minimum face size check
            if fw < 30 or fh < 30:
                return "Unknown", 0.0
                
            face_crop = person_crop[fy:fy+fh, fx:fx+fw]
            try:
                # Using opencv backend within deepface for much faster real-time processing
                # set enforce_detection=False since we already verified the face crop manually
                result = DeepFace.analyze(face_crop, 
                                          actions=['emotion'], 
                                          enforce_detection=False, 
                                          detector_backend='opencv', 
                                          silent=True)
                                          
                # Handle multiple faces if returned as a list
                if isinstance(result, list):
                    result = result[0]
                    
                dominant_emotion = result['dominant_emotion'].capitalize()
                score = result['emotion'][result['dominant_emotion']]
                return dominant_emotion, score
            except Exception:
                # Fallback for any unexpected errors
                return "Unknown", 0.0
        else:
            return "Unknown", 0.0
