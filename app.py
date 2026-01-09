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
# Custom CSS
# -------------------------------
st.markdown("""
<style>
.title {
    font-size: 48px;
    font-weight: bold;
    color: #ffffff;
    text-shadow: 2px 2px 8px black;
}
.subtitle {
    font-size: 18px;
    color: #e0f2f1;
}
.card {
    background: rgba(255,255,255,0.9);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.2);
}
.badge {
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: bold;
    color: white;
    display: inline-block;
}
.hazard { background:#e63946; }
.recycle { background:#2a9d8f; }
.recovery { background:#f4a261; }
.landfill { background:#6c757d; }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Header
# -------------------------------
st.markdown("<div class='title'>♻ Smart Waste AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered waste identification for Smart Cities</div>", unsafe_allow_html=True)
st.markdown("---")

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
left, right = st.columns([1,1])

with left:
    st.markdown("## 📤 Upload Waste Image")
    uploaded = st.file_uploader("", type=["jpg","png","jpeg"])

with right:
    st.markdown("## 🧠 AI Prediction")

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_resized = img.resize((224,224))

        arr = np.expand_dims(np.array(img_resized)/255.0, axis=0)

        pred = model.predict(arr)
        idx = np.argmax(pred)
        confidence = float(np.max(pred)*100)

        label = class_names[idx]
        rule = waste_rules[label]

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(img, width=300)

        st.markdown(f"### **{label.upper()}**")
        st.progress(confidence/100)
        st.write(f"Confidence: **{confidence:.2f}%**")

        st.markdown(f"<span class='badge {rule['badge']}'>{rule['type']}</span>", unsafe_allow_html=True)
        st.write(rule["message"])
        st.success(f"Action: {rule['action']}")

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("Upload an image to start AI waste classification")
