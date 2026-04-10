# Human Detection and Emotion Tracking

This project is a Python-based AI application that uses Computer Vision and Deep Learning to process live webcam or recorded video feeds. It detects humans, tracks them consistently across frames, recognizes their facial emotions, and logs the data to generate interactive emotion-time graphs.

## Features
- **Human Detection**: Powered by YOLOv8 (`ultralytics`).
- **Object Tracking**: Maintains consistent IDs across frames using DeepSORT (`deep-sort-realtime`).
- **Emotion Recognition**: Predicts facial emotions (Happy, Sad, Angry, Fear, Surprise, Neutral, Disgust) using a robust CNN model via `deepface`.
- **Data Logging**: Records `(PersonID, Timestamp, Emotion, Confidence)` to a CSV file.
- **Interactive Visualization**: Automatically generates offline interactive HTML graphs using `plotly`, with clickable data points that show the exact snapshot of the person's face at that moment.

## Setup Instructions

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Note: OpenCV and TensorFlow will be installed as dependencies of `deepface` and `ultralytics`.)*

## Running the Application

Execute the main script:
```bash
python main.py
```

You will be prompted to select your input mode:
```
========================================
Human Detection and Emotion Recognition
========================================
Select Input Mode:
1 -> Live Webcam
2 -> Recorded Video File

Enter your choice (1 or 2): 
```

- If you select Option 1, your default webcam will activate.
- If you select Option 2, provide the absolute or relative path to your `.mp4` or video file.

Press **`q`** at any time while the video window is focused to stop processing. 

### Outputs
Once execution stops, the script will automatically generate:
1. `emotion_log.csv`: Raw timeline data.
2. `emotion_graphs.html`: An interactive Playly graph. Open this file in your web browser.
3. `snapshots/`: A folder containing cropped images of faces saved at calculation intervals, which are linked to the interactive graph tooltips.
