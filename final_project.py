import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Set Streamlit page config
st.set_page_config(
    page_title="Cognitive Animal Predictor",
    page_icon="🐾",
    layout="centered",
)

# Custom CSS for dark mode and styling
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: white;
        }
        .stApp {
            background: linear-gradient(135deg, #1a1a1a, #252525);
        }
        .title {
            text-align: center;
            font-size: 50px;
            font-weight: bold;
            color: #ffcc00;
            text-shadow: 2px 2px 10px rgba(255, 204, 0, 0.7);
            animation: fadeIn 2s;
        }
        .subtitle {
            text-align: center;
            font-size: 22px;
            opacity: 0.8;
        }
        .description {
            text-align: center;
            font-size: 18px;
            margin-top: 10px;
            color: #ccc;
        }
        .upload-container {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        .upload-box {
            border: 2px dashed #ffcc00;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            width: 80%;
            max-width: 500px;
            color: white;
            font-size: 16px;
            transition: 0.3s;
        }
        .upload-box:hover {
            background: rgba(255, 204, 0, 0.1);
        }
        .predict-button {
            display: block;
            margin: 20px auto;
            background: #ffcc00;
            color: black;
            padding: 12px 24px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            box-shadow: 2px 2px 10px rgba(255,204,0,0.5);
            transition: 0.3s;
            text-align: center;
            border: none;
        }
        .predict-button:hover {
            background: #ff9900;
            transform: scale(1.1);
        }
        .confidence {
            text-align: center;
            font-size: 18px;
            color: #ffcc00;
            font-weight: bold;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('animals_classifier.hdf5')
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# Landing Page Content
st.markdown('<h1 class="title">🐾 Cognitively Advanced Animals Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Animal Intelligence Classification</p>', unsafe_allow_html=True)
st.markdown('<p class="description">Upload an image of an elephant, crow, rat, or bear, and let AI determine its cognitive ability!</p>', unsafe_allow_html=True)

# Upload Button
file = st.file_uploader("Upload an image", type=["jpg", "png"])

# Prediction Function
def import_and_predict(image_data, model):
    try:
        size = (224, 224)  # Ensure this matches your model's expected input size
        image = ImageOps.fit(image_data, size, Image.LANCZOS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis, ...]  # Add batch dimension
        prediction = model.predict(img_reshape)
        return prediction
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        return None

if file is not None:
    image = Image.open(file)
    st.image(image, use_container_width=True, caption="📷 Uploaded Image")
    
    if model is not None:
        prediction = import_and_predict(image, model)
        
        if prediction is not None:
            class_names = ['bears', 'crows', 'elephants', 'rats']
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100  # Convert to percentage
            
            # Display Prediction
            st.markdown(f'<p class="confidence">🔍 Predicted: <b>{predicted_class}</b> ({confidence:.2f}% confidence)</p>', unsafe_allow_html=True)
        else:
            st.error("❌ Prediction failed. Please check the input and try again.")
    else:
        st.error("❌ Model is not loaded. Please check the error messages above.")

# Footer
st.markdown("<br><hr><p style='text-align:center; color:#ffcc00;'>© 2023 Cognitively Advanced Animals - AI Prediction Model</p>", unsafe_allow_html=True)
