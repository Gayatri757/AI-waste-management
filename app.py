import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ---------------------------------------
# Page Config
# ---------------------------------------
st.set_page_config(page_title="Smart Waste AI", layout="wide")

# ---------------------------------------
# Custom CSS for UI
# ---------------------------------------
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1rem;}

.stApp {
    background: linear-gradient(120deg,#ecfeff,#d1fae5);
}

.card {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 35px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.12);
}

.title {
    font-size: 48px;
    font-weight: 900;
    color: #064e3b;
}

.subtitle {
    color: #065f46;
    font-size: 18px;
}

.result {
    background: linear-gradient(135deg,#d1fae5,#ecfeff);
    padding: 22px;
    border-radius: 15px;
    margin-top: 20px;
    font-size: 20px;
    font-weight: 700;
    color: #064e3b;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------
# Load Model
# ---------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("waste_classifier_mobilenet.h5")

model = load_model()

# ---------------------------------------
# Class Names
# ---------------------------------------
class_names = ['battery','biological','cardboard','clothes','glass',
               'metal','paper','plastic','shoes','trash']

# ---------------------------------------
# Smart City Waste Rules
# ---------------------------------------
waste_rules = {
    # E-WASTE
    "battery": ("E-Waste","⚠ Electronic waste — Inform supervisor"),
    "metal": ("E-Waste","⚠ Possible electronic waste — Inform supervisor"),

    # WET WASTE
    "biological": ("Wet Waste","🌱 Compost / green bin"),

    # DRY WASTE (Recyclable)
    "plastic": ("Dry Waste","♻ Blue recycling bin"),
    "glass": ("Dry Waste","🍾 Glass recycling bin"),
    "paper": ("Dry Waste","📄 Paper recycling bin"),
    "cardboard": ("Dry Waste","📦 Cardboard recycling bin"),

    # DRY WASTE (Non-recyclable)
    "clothes": ("Dry Waste","🟨 Cloth recovery / donation bin"),
    "shoes": ("Dry Waste","⬛ Landfill or reuse bin"),

    # MIXED / DIRTY
    "trash": ("Landfill","⬛ Black landfill bin")
}

# ---------------------------------------
# Initialize Stats
# ---------------------------------------
if "stats" not in st.session_state:
    st.session_state.stats = {
        "total":0,
        "wet":0,
        "dry":0,
        "hazardous":0,
        "landfill":0
    }

if "last_file" not in st.session_state:
    st.session_state.last_file = None

# ---------------------------------------
# Display Metrics
# ---------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Waste", st.session_state.stats["total"])
c2.metric("Wet Waste", st.session_state.stats["wet"])
c3.metric("Dry Waste", st.session_state.stats["dry"])
c4.metric("E-Waste", st.session_state.stats["hazardous"])
c5.metric("Landfill", st.session_state.stats["landfill"])

# ---------------------------------------
# Main Card UI
# ---------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='title'>♻ Smart Waste AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered waste classification for Smart Cities</div><br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader("📤 Upload Waste Image", type=["jpg","png","jpeg"])

with col2:
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_resized = img.resize((224,224))
        arr = np.expand_dims(np.array(img_resized)/255.0, axis=0)

        # Prediction
        pred = model.predict(arr)
        idx = np.argmax(pred)
        confidence = float(np.max(pred)*100)
        label = class_names[idx]

        # Get waste type & action
        waste_type, action = waste_rules[label]

        # Smart E-Waste override
        if label in ["battery","metal"] and confidence < 85:
            waste_type = "E-Waste"
            action = "⚠ Electronic waste detected — Inform supervisor"

        # Update stats if new file
        if uploaded != st.session_state.last_file:
            st.session_state.stats["total"] += 1

            if waste_type == "Wet Waste":
                st.session_state.stats["wet"] += 1
            elif waste_type == "Dry Waste":
                st.session_state.stats["dry"] += 1
            elif waste_type == "E-Waste":
                st.session_state.stats["hazardous"] += 1
            else:
                st.session_state.stats["landfill"] += 1

            st.session_state.last_file = uploaded

        # Display image and result
        st.image(img, width=260)
        st.markdown(
            f"<div class='result'>🗂 {label.upper()} ({confidence:.2f}%)<br>🚮 {action}</div>",
            unsafe_allow_html=True
        )

        # Play optional success audio safely
        audio_file = "success.mp3"  # make sure this file is in same folder as app.py
        if os.path.exists(audio_file):
            st.audio(audio_file)
        else:
            st.warning("Audio file not found — skipping sound")

    else:
        st.info("Upload an image to get waste classification")

st.markdown("</div>", unsafe_allow_html=True)
