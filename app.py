import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load your Keras model
try:
    model = load_model('model.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")

# Function to preprocess the image/frame and make predictions
def preprocess_image(image):
    image = image.resize((300, 300))  # Example resize, adjust as needed
    image_array = np.array(image) / 255.0  # Normalize if needed
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit app
st.title("Can Classifier")
st.write("This app classifies cans as defective or non-defective.")

mode = st.radio("Choose a mode:", ('Real-Time Classification', 'Upload Picture'))

class_names = ["Non-Defective", "Defective"]  # Adjust according to your model's classes

if mode == 'Real-Time Classification':
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture frame from camera. Please check your camera device.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prediction = predict(Image.fromarray(frame))
        result = class_names[np.argmax(prediction)]

        # Display the result on the frame
        cv2.putText(frame, f"Prediction: {result}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        FRAME_WINDOW.image(frame)

    cap.release()

elif mode == 'Upload Picture':
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        prediction = predict(image)
        result = class_names[np.argmax(prediction)]

        st.write(f"The can is **{result}**.")
