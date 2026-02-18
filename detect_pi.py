import cv2
import numpy as np
import tensorflow as tf


MODEL_PATH = "pothole_model.tflite"
VIDEO_SOURCE = "road.mp4"            
IMG_SIZE = 224
THRESHOLD = 0.8            
CONFIRM_FRAMES = 5          


interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite model loaded")

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("Could not open camera/video source")
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
    img = np.expand_dims(img, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0][0]

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

    cv2.imshow("Pothole Detection - Raspberry Pi (TFLite)", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
