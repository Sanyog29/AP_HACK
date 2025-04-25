from flask import Flask, Response, render_template_string, request, jsonify
import cv2
import os
import numpy as np
from ultralytics import YOLO
import time

app = Flask(__name__)

# Configuration
MODEL_PATH = os.path.join(os.getcwd(), 'model', 'best.pt')
# Fallback to pretrained model if custom model doesn't exist
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'yolov8n.pt'
    print(f"Custom model not found at {MODEL_PATH}, using pretrained model")
else:
    print(f"Using custom model: {MODEL_PATH}")

# Initialize model
model = YOLO(MODEL_PATH)
gunny_bag_class_id = 0  # For custom model, gunny_bag is class 0

# Global variables for statistics
total_bags_counted = 0
detection_history = []
current_video_source = 'webcam'  # Default to webcam

# Lightweight contrast enhancement
def enhance_contrast(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

def process_frame(frame):
    global total_bags_counted, detection_history
    
    # Apply contrast enhancement
    enhanced = enhance_contrast(frame)
    
    # Run inference with tracking
    results = model.track(enhanced, persist=True, tracker="bytetrack.yaml")
    
    # Count gunny bags
    current_count = sum(1 for r in results[0].boxes if int(r.cls) == gunny_bag_class_id)
    
    # Update statistics
    timestamp = time.time()
    detection_history.append((timestamp, current_count))
    
    # Keep only last 100 detections in history
    if len(detection_history) > 100:
        detection_history.pop(0)
    
    # Get the annotated frame
    annotated_frame = results[0].plot()
    
    # Add count text
    cv2.putText(
        annotated_frame, 
        f"Gunny Bags: {current_count}", 
        (20, 40), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 0), 
        2
    )
    
    return annotated_frame, current_count

def get_video_source():
    global current_video_source
    
    if current_video_source == 'webcam':
        return cv2.VideoCapture(0)
    elif current_video_source == 'rtsp':
        return cv2.VideoCapture('rtsp://your_cctv_feed')  # Replace with actual RTSP URL
    else:
        # Assume it's a video file
        video_path = os.path.join(os.getcwd(), current_video_source)
        return cv2.VideoCapture(video_path)

def generate_frames():
    cap = get_video_source()
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            processed, _ = process_frame(frame)
            _, buffer = cv2.imencode('.jpg', processed)
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()

@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Gunny Bag Tracker</title>
            <style>
                body { 
                    margin: 0; 
                    background: #1a1a1a; 
                    color: white;
                    font-family: Arial, sans-serif;
                }
                .container {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    padding: 20px;
                }
                #video { 
                    max-width: 100%; 
                    border: 2px solid #444;
                    border-radius: 5px;
                }
                #counter {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: rgba(0,0,0,0.7);
                    color: #fff;
                    padding: 15px;
                    border-radius: 5px;
                    font-family: Arial;
                    z-index: 100;
                }
                .controls {
                    margin-top: 20px;
                    background: #333;
                    padding: 15px;
                    border-radius: 5px;
                    width: 80%;
                    max-width: 800px;
                }
                .controls h3 {
                    margin-top: 0;
                }
                select, button {
                    padding: 8px;
                    margin: 5px;
                    background: #444;
                    color: white;
                    border: none;
                    border-radius: 3px;
                }
                button {
                    cursor: pointer;
                }
                button:hover {
                    background: #555;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Gunny Bag Detection System</h1>
                <div id="counter">Count: 0</div>
                <img id="video" src="/video_feed">
                
                <div class="controls">
                    <h3>Video Source</h3>
                    <select id="videoSource">
                        <option value="webcam">Webcam</option>
                        <option value="rtsp">RTSP Stream</option>
                        <option value="sample.mp4">Sample Video</option>
                    </select>
                    <button id="changeSource">Change Source</button>
                </div>
            </div>
            
            <script>
                const counter = document.getElementById('counter');
                const eventSource = new EventSource('/count_updates');
                const videoSource = document.getElementById('videoSource');
                const changeSourceBtn = document.getElementById('changeSource');
                
                eventSource.onmessage = (e) => {
                    counter.textContent = `Count: ${e.data}`;
                };
                
                changeSourceBtn.addEventListener('click', () => {
                    fetch('/change_source', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            source: videoSource.value
                        })
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            // Reload the video feed
                            const videoElement = document.getElementById('video');
                            videoElement.src = '/video_feed?' + new Date().getTime();
                        }
                    });
                });
            </script>
        </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/count_updates')
def count_updates():
    def generate_counts():
        while True:
            if detection_history:
                _, count = detection_history[-1]
                yield f"data: {count}\n\n"
            else:
                yield f"data: 0\n\n"
            time.sleep(0.5)
    return Response(generate_counts(), mimetype='text/event-stream')

@app.route('/change_source', methods=['POST'])
def change_source():
    global current_video_source
    data = request.get_json()
    current_video_source = data.get('source', 'webcam')
    return jsonify({'success': True, 'source': current_video_source})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)