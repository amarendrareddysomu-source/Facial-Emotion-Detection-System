import cv2
import time
import os
import csv
from datetime import timedelta
from collections import deque, Counter

# Local module imports
from human_detector import HumanDetector
from tracker import HumanTracker
from emotion_model import EmotionDetector
from graph_visualization import generate_graphs

def run_pipeline(video_path, is_live):
    detector = HumanDetector()
    tracker = HumanTracker(max_age=30)
    emotion_detector = EmotionDetector()

    cap = cv2.VideoCapture(video_path)
    
    # Robust fallback for Mac OS live webcams
    if is_live and not cap.isOpened():
        print(f"Warning: Camera index {video_path} failed. Searching for other connected cameras...")
        for i in range(1, 4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Success! Found active camera at index {i}.")
                break
                
    if not cap.isOpened():
        print(f"\nError: Unable to open video source!")
        if is_live:
            print("Troubleshooting Mac OS Webcams:")
            print("1. Ensure your terminal (or IDE) has 'Camera' permissions in Mac System Preferences -> Security & Privacy.")
            print("2. Unplug any external webcams and try again.")
        return

    # Prepare logging and snapshot directories
    os.makedirs('snapshots', exist_ok=True)
    log_file = 'emotion_log.csv'
    
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['PersonID', 'Timestamp', 'Emotion', 'Confidence', 'SnapshotPath'])

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 30.0

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if orig_w > 0 and orig_h > 0:
        cv2.namedWindow("Emotion Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Emotion Tracking", orig_w, orig_h)

    frame_count = 0
    start_time = time.time()
    
    # Emotion caching (Memoization) to circumvent real-time bottlenecks
    emotion_cache = {}
    
    emotion_history = {}
    def smooth_emotion(person_id, new_emotion, window=10):
        if person_id not in emotion_history:
            emotion_history[person_id] = deque(maxlen=window)
        emotion_history[person_id].append(new_emotion)
        return Counter(emotion_history[person_id]).most_common(1)[0][0]
    
    EMOTION_UPDATE_INTERVAL = 10 

    print("Processing stream. Press 'q' to stop early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Use wall-clock time if live stream, or frame-calculated time if from a file.
        if is_live:
            current_time_sec = time.time() - start_time
        else:
            current_time_sec = frame_count / fps
            
        timestamp_str = str(timedelta(seconds=int(current_time_sec)))

        # 1. Object Identification
        person_detections, ignored_objects = detector.detect(frame)
            
        # Draw ignored objects to prove they were detected but ignored for tracking
        for bbox, conf, label in ignored_objects:
            ix1, iy1, ix2, iy2 = map(int, bbox)
            
            if label.startswith("Other ("):
                display_label = label[7:-1]
                color = (0, 170, 255)
            else:
                display_label = label
                color = (255, 204, 0)
            
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)  
            cv2.putText(frame, display_label, (ix1 + 5, iy1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 2. Tracking (Only process humans)
        tracks = tracker.update(person_detections, frame)

        # 3. Emotion Detection & Logging
        for track_id, ltrb in tracks:
            x1, y1, x2, y2 = map(int, ltrb)
            
            cache_entry = emotion_cache.get(track_id)
            
            if cache_entry is None or (frame_count - cache_entry['last_update_frame']) >= EMOTION_UPDATE_INTERVAL:
                raw_emotion, raw_conf = emotion_detector.detect_emotion(frame, ltrb)
                
                # Smooth emotion over last 10 frames to avoid flickering
                emotion = smooth_emotion(track_id, raw_emotion)
                
                emotion_cache[track_id] = {
                    'emotion': emotion,
                    'conf': raw_conf,
                    'last_update_frame': frame_count
                }
                
                snapshot_filename = f"snapshots/p{track_id}_{frame_count}.jpg"
                
                h, w, _ = frame.shape
                pad = 10
                crop_y1, crop_y2 = max(0, y1-pad), min(h, y2+pad)
                crop_x1, crop_x2 = max(0, x1-pad), min(w, x2+pad)
                person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                
                if person_crop.size > 0:
                    cv2.imwrite(snapshot_filename, person_crop)
                else:
                    snapshot_filename = ""
                
                with open(log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([track_id, timestamp_str, emotion, raw_conf, snapshot_filename])
            else:
                emotion = cache_entry['emotion']
                raw_conf = cache_entry['conf']

            # Visualization on the preview frame array
            human_color = (68, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), human_color, 2)
            
            # Skip drawing the specific emotion text if it returned Unknown due to missing/small face
            if emotion == "Unknown":
                label = f"Person {track_id}"
            else:
                label = f"Person {track_id} | {emotion} | {raw_conf:.0f}%"
                
            cv2.putText(frame, label, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, human_color, 2)

        # Show frame physically in a cv2 window
        cv2.imshow('Emotion Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Generating Interactive Emotion Graph...")
    # Invoke our plotly generator to serialize output HTML
    generate_graphs(log_file, "emotion_graphs.html")
    print("Execution Finished! Rendered outputs to emotion_graphs.html")


def main():
    print("=" * 40)
    print("Human Detection and Emotion Recognition")
    print("=" * 40)
    print("Select Input Mode:")
    print("1 -> Live Webcam")
    print("2 -> Recorded Video File")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == '1':
        # VideoCapture(0) represents default local camera node
        run_pipeline(0, is_live=True)
    elif choice == '2':
        video_path = input("Enter video file path: ").strip()
        if not os.path.exists(video_path):
            print("Video file not found or path invalid. Exit.")
            return
        run_pipeline(video_path, is_live=False)
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
