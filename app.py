import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from PIL import Image
import tensorflow as tf  # Add this import

# Define optimizer
optimizer = Adam(learning_rate=0.001)

# Compile the model
# Load your Keras model
try:
    model = load_model('model.h5')
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
except Exception as e:
    st.error(f"Error loading model: {e}")

@@ -27,6 +23,15 @@ def predict(image):
    prediction = model.predict(processed_image)
    return prediction

# Function to preprocess the image and make predictions
def preprocess_and_predict(image):
    image = image.resize((300, 300))  # Adjust target_size as needed
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    result = 'Defective' if prediction[0][0] > 0.5 else 'Non-Defective'  # Adjust the condition as needed
    return result

# Streamlit app
st.title("Can Classifier")
st.write("This app classifies cans as defective or non-defective.")
@@ -53,7 +58,7 @@ def predict(image):
    else:
        st.info(f"Found {num_cameras} camera(s). Using camera index 0.")

    cap = cv2.VideoCapture(-1)
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
@@ -81,7 +86,6 @@ def predict(image):
        st.write("")
        st.write("Classifying...")

        prediction = predict(image)
        result = class_names[np.argmax(prediction)]
        result = preprocess_and_predict(image)

        st.write(f"The can is **{result}**.")
