import cv2
import numpy as np
import tensorflow as tf


MODEL_PATH = "pothole_model_tf214.h5"
VIDEO_SOURCE = "test.mp4"   # Use 0 for webcam, or "road.mp4" for video file
IMG_SIZE = 224
THRESHOLD = 0.8             
CONFIRM_FRAMES = 5          

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded")

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("Could not open video/camera")
    exit()

print("Video started. Press Q to quit.")

pothole_count = 0  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    roi = frame[int(h * 0.5):h, 0:w]

    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0][0]

    if pred > THRESHOLD:
        pothole_count += 1
    else:
        pothole_count = 0

    if pothole_count >= CONFIRM_FRAMES:
        text = f"POTHOLE ({pred:.2f})"
        color = (0, 0, 255)  # Red
    else:
        text = f"NORMAL ROAD ({pred:.2f})"
        color = (0, 255, 0)  # Green

    cv2.rectangle(frame, (0, int(h * 0.5)), (w, h), (255, 0, 0), 2)

    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Pothole Detection (ML)", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

