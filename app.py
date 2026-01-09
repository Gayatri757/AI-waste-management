import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import base64

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Smart Waste AI", layout="wide")

# -------------------------------
# Background Image
# -------------------------------
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("background.png")

# -------------------------------
# Dark Overlay + Glass UI
# -------------------------------
st.markdown("""
<style>
.stApp {
    background-image: url("background.png");
    background-size: cover;
    background-position: center;
}

/* Dark overlay */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.55);
    z-index: -1;
}

/* Glass container */
.glass {
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 40px;
    box-shadow: 0 0 40px rgba(0,0,0,0.4);
}

/* Title */
.title {
    font-size: 50px;
    font-weight: 800;
    color: white;
}
.subtitle {
    color: #d1fae5;
    font-size: 18px;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg,#10b981,#059669);
    color:white;
    border:none;
    border-radius: 10px;
    padding: 12px 25px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("waste_classifier_mobilenet.h5")

model = load_model()

# -------------------------------
# Labels
# -------------------------------
class_names = ['battery','biological','cardboard','clothes','glass',
               'metal','paper','plastic','shoes','trash']

# -------------------------------
# Waste Rules
# -------------------------------
waste_rules = {
    "battery": {"type":"Hazardous","badge":"hazard","message":"⚠ Battery waste is dangerous.","action":"Send to hazardous waste unit"},
    "biological": {"type":"Organic","badge":"recycle","message":"🌱 Biodegradable waste detected.","action":"Send to compost"},
    "plastic": {"type":"Recyclable","badge":"recycle","message":"♻ Plastic can be recycled.","action":"Send to plastic recycling"},
    "glass": {"type":"Recyclable","badge":"recycle","message":"♻ Glass can be recycled.","action":"Send to glass recycling"},
    "metal": {"type":"Recovery","badge":"recovery","message":"💰 Metal has recovery value.","action":"Send for metal recovery"},
    "paper": {"type":"Recyclable","badge":"recycle","message":"📄 Paper detected.","action":"Send to paper recycling"},
    "cardboard": {"type":"Recyclable","badge":"recycle","message":"📦 Cardboard detected.","action":"Send to cardboard recycling"},
    "shoes": {"type":"Landfill","badge":"landfill","message":"👟 Shoes are not recyclable.","action":"Send to landfill"},
    "clothes": {"type":"Recovery","badge":"recovery","message":"👕 Clothes can be reused.","action":"Send to donation / recovery center"},
    "trash": {"type":"Landfill","badge":"landfill","message":"🗑 General waste.","action":"Dispose safely"}
}

# -------------------------------
# Layout
# -------------------------------
st.markdown("<div class='glass'>", unsafe_allow_html=True)

st.markdown("<div class='title'>♻ Smart Waste AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered waste classification for Smart Cities</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Waste Image")
    uploaded = st.file_uploader("", type=["jpg","png","jpeg"])

with col2:
    st.subheader("🧠 AI Prediction")

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_resized = img.resize((224,224))
        arr = np.expand_dims(np.array(img_resized)/255.0, axis=0)

        pred = model.predict(arr)
        idx = np.argmax(pred)
        confidence = float(np.max(pred)*100)

        label = class_names[idx]
        rule = waste_rules[label]

        st.image(img, width=250)
        st.success(f"**{label.upper()}**  ({confidence:.2f}%)")
        st.info(rule["message"])
        st.warning(f"Action: {rule['action']}")

    else:
        st.info("Upload an image to get AI result")

st.markdown("</div>", unsafe_allow_html=True)


