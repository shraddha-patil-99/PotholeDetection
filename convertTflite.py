import tensorflow as tf

print("TensorFlow version:", tf.__version__)

model = tf.keras.models.load_model("pothole_model_tf214.h5", compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("pothole_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as pothole_model.tflite")




