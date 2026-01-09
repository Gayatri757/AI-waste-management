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
# Custom CSS
# -------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #f6f9fc, #e9f5ec);
}
.title {
    font-size: 42px;
    font-weight: bold;
    color: #2b7a78;
}
.subtitle {
    font-size: 18px;
    color: #555;
}
.card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.08);
}
.badge {
    padding: 8px 15px;
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
st.markdown("<div class='subtitle'>AI-powered waste identification & disposal guidance</div>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "waste_classifier_mobilenet.h5")
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -------------------------------
# Labels
# -------------------------------
class_names = ['battery','biological','cardboard','clothes','glass',
               'metal','paper','plastic','shoes','trash']

# -------------------------------
# Knowledge Base
# -------------------------------
waste_rules = {
    "battery": {"type":"Hazardous","badge":"hazard","message":"⚠ Battery is dangerous. Do not touch with bare hands.","action":"Send to hazardous waste bin"},
    "biological": {"type":"Organic","badge":"recycle","message":"🥬 Biodegradable waste.","action":"Send to compost unit"},
    "plastic": {"type":"Recyclable","badge":"recycle","message":"♻ Plastic can be recycled.","action":"Send to plastic recycling"},
    "glass": {"type":"Recyclable","badge":"recycle","message":"♻ Glass is recyclable.","action":"Send to glass recycling"},
    "metal": {"type":"Valuable","badge":"recovery","message":"💰 Metal can be recovered and reused.","action":"Send for metal recovery"},
    "paper": {"type":"Recyclable","badge":"recycle","message":"📄 Paper detected.","action":"Send to paper recycling"},
    "cardboard": {"type":"Recyclable","badge":"recycle","message":"📦 Cardboard detected.","action":"Send to cardboard recycling"},
    "shoes": {"type":"Trash","badge":"landfill","message":"👟 Shoes are not recyclable.","action":"Send to landfill"},
    "clothes": {"type":"Recovery","badge":"recovery","message":"👕 Clothes can be reused or donated.","action":"Send to recovery center"},
    "trash": {"type":"Landfill","badge":"landfill","message":"🗑 General waste.","action":"Dispose safely"}
}

# -------------------------------
# Layout
# -------------------------------
left, right = st.columns([1,1])

with left:
    st.markdown("### 📤 Upload Waste Image")
    uploaded = st.file_uploader("", type=["jpg","png","jpeg"])

with right:
    st.markdown("### 🧠 AI Prediction")

    if uploaded:
        img = Image.open(uploaded).convert("RGB").resize((224,224))
        arr = np.expand_dims(np.array(img)/255.0, axis=0)

        pred = model.predict(arr)
        idx = np.argmax(pred)
        confidence = float(np.max(pred)*100)

        label = class_names[idx]
        rule = waste_rules[label]

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(img, width=250)

        st.markdown(f"### **{label.upper()}**")
        st.progress(confidence/100)
        st.write(f"Confidence: **{confidence:.2f}%**")

        st.markdown(f"<span class='badge {rule['badge']}'>{rule['type']}</span>", unsafe_allow_html=True)
        st.write(rule["message"])
        st.success(f"Action: {rule['action']}")

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("Upload an image to start detection")
