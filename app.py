import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import streamlit as st
from PIL import Image
import cv2

def predict_image(filepath):
    model = load_model('model.h5')  # Pastikan path ini sesuai dengan lokasi model Anda
    img = image.load_img(filepath, target_size=(300, 300))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    if prediction[0][0] <= 0.5:
        return 'Cacat'
    else:
        return 'Normal'

# Streamlit app
st.title("Can Classifier")
st.write("This app classifies cans as defective or non-defective.")

mode = st.radio("Choose a mode:", ('Real-Time Classification', 'Upload Picture'))

class_names = ["Non-Defective", "Defective"]  # Adjust according to your model's classes

if mode == 'Real-Time Classification':
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])

    # Check available cameras
    num_cameras = 0
    while True:
        cap = cv2.VideoCapture(num_cameras)
        if not cap.isOpened():
            break
        cap.release()
        num_cameras += 1

    if num_cameras == 0:
        st.error("No camera detected.")
    else:
        st.info(f"Found {num_cameras} camera(s). Using camera index 0.")

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

st.title('Klasifikasi Kaleng: Cacat atau Tidak')

mode = st.sidebar.selectbox("Mode", ["Upload Picture"])

if mode == 'Upload Picture':
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Simpan gambar yang diunggah ke file sementara
        temp_filepath = 'temp_image.png'
        img.save(temp_filepath)

        # Prediksi gambar
        result = predict_image(temp_filepath)

        st.write(f"The can is {result}.")
