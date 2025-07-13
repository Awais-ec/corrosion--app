import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the TFLite model using TensorFlow
interpreter = tf.lite.Interpreter(model_path="corrosion_model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# App title and instructions
st.title("🛠️ Pipeline Corrosion Detection (Lite Model)")
st.write("Upload a pipeline image to check for **Corrosion**.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((128, 128))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Prepare image
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Make prediction
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    # Display result
    if prediction > 0.5:
        st.error(f"✅ No Corrosion Detected! (Confidence: {prediction:.2f})")
    else:
        st.success( f"🚨 Corrosion Detected. (Confidence: {1 - prediction:.2f})")
