import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Smart Waste AI", layout="wide")

# -------------------------------
# Modern Glass UI (No Background Image)
# -------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(120deg,#ecfeff,#d1fae5);
}

/* Glass container */
.glass {
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 40px;
    box-shadow: 0 0 40px rgba(0,0,0,0.1);
}

/* Title */
.title {
    font-size: 48px;
    font-weight: 800;
    color: #064e3b;
}
.subtitle {
    color: #065f46;
    font-size: 18px;
}

/* Loader */
.loader {
    border: 6px solid rgba(0,0,0,0.1);
    border-top: 6px solid #10b981;
    border-radius: 50%;
    width: 60px;
    height: 60px;
    animation: spin 1s linear infinite;
    margin:auto;
}
@keyframes spin {
    0% {transform: rotate(0deg);}
    100% {transform: rotate(360deg);}
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
    "battery": {"type":"Hazardous","message":"⚠ Battery waste is dangerous.","action":"Send to hazardous waste unit"},
    "biological": {"type":"Organic","message":"🌱 Biodegradable waste detected.","action":"Send to compost"},
    "plastic": {"type":"Recyclable","message":"♻ Plastic can be recycled.","action":"Send to plastic recycling"},
    "glass": {"type":"Recyclable","message":"♻ Glass can be recycled.","action":"Send to glass recycling"},
    "metal": {"type":"Recovery","message":"💰 Metal has recovery value.","action":"Send for metal recovery"},
    "paper": {"type":"Recyclable","message":"📄 Paper detected.","action":"Send to paper recycling"},
    "cardboard": {"type":"Recyclable","message":"📦 Cardboard detected.","action":"Send to cardboard recycling"},
    "shoes": {"type":"Landfill","message":"👟 Shoes are not recyclable.","action":"Send to landfill"},
    "clothes": {"type":"Recovery","message":"👕 Clothes can be reused.","action":"Send to donation / recovery center"},
    "trash": {"type":"Landfill","message":"🗑 General waste.","action":"Dispose safely"}
}

# -------------------------------
# Smart City Stats
# -------------------------------
if "stats" not in st.session_state:
    st.session_state.stats = {
        "total": 0,
        "recyclable": 0,
        "hazardous": 0,
        "recovered": 0
    }

# -------------------------------
# Dashboard
# -------------------------------
st.markdown("## 🌍 Smart City Waste Monitor")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Waste", st.session_state.stats["total"])
c2.metric("Recyclable", st.session_state.stats["recyclable"])
c3.metric("Hazardous", st.session_state.stats["hazardous"])
c4.metric("Recovered", st.session_state.stats["recovered"])

# -------------------------------
# Main Glass UI
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

        with st.spinner("AI scanning waste..."):
            st.markdown("<div class='loader'></div>", unsafe_allow_html=True)
            pred = model.predict(arr)

        idx = np.argmax(pred)
        confidence = float(np.max(pred)*100)

        label = class_names[idx]
        rule = waste_rules[label]

        # Update stats
        st.session_state.stats["total"] += 1
        if rule["type"] == "Hazardous":
            st.session_state.stats["hazardous"] += 1
            st.audio("sounds/alert.mp3")
        elif rule["type"] in ["Recyclable", "Organic"]:
            st.session_state.stats["recyclable"] += 1
            st.audio("sounds/success.mp3")
        elif rule["type"] == "Recovery":
            st.session_state.stats["recovered"] += 1
            st.audio("sounds/success.mp3")

        st.image(img, width=250)
        st.success(f"**{label.upper()}**  ({confidence:.2f}%)")
        st.info(rule["message"])
        st.warning(f"Action: {rule['action']}")

    else:
        st.info("Upload an image to get AI result")

st.markdown("</div>", unsafe_allow_html=True)
