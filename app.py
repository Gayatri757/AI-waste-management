import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# -------------------------------
# UI SETTINGS
# -------------------------------
st.set_page_config(page_title="Smart Waste AI", layout="centered")

st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
.big-font {
    font-size:30px !important;
    font-weight: bold;
}
.card {
    background-color:white;
    padding:20px;
    border-radius:10px;
    box-shadow: 0px 0px 10px #ddd;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">♻ AI Based Smart Waste Management System</p>', unsafe_allow_html=True)
st.write("Upload a waste image to get disposal & safety recommendation")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("waste_classifier.h5")
    return model

model = load_model()

# -------------------------------
# CLASS LABELS (same order used during training)
# -------------------------------
class_names = ['battery','biological','cardboard','clothes','glass',
               'metal','paper','plastic','shoes','trash']

# -------------------------------
# KNOWLEDGE BASE
# -------------------------------
waste_rules = {
    "battery": {"type":"Hazardous","message":"⚠ Battery is hazardous. Inform supervisor.","action":"Send to hazardous waste bin"},
    "biological": {"type":"Organic","message":"🥬 Biodegradable waste.","action":"Send to compost unit"},
    "plastic": {"type":"Recyclable","message":"♻ Plastic detected.","action":"Send to recycling"},
    "glass": {"type":"Recyclable","message":"♻ Glass detected.","action":"Send to glass recycling"},
    "metal": {"type":"Valuable","message":"💰 Metal has recovery value.","action":"Send for metal recovery"},
    "paper": {"type":"Recyclable","message":"📄 Paper detected.","action":"Send to paper recycling"},
    "cardboard": {"type":"Recyclable","message":"📦 Cardboard detected.","action":"Send to cardboard recycling"},
    "shoes": {"type":"Trash","message":"👟 Shoes are non-recyclable.","action":"Send to landfill"},
    "clothes": {"type":"Recovery","message":"👕 Clothes can be reused or donated.","action":"Send to recovery center"},
    "trash": {"type":"Landfill","message":"🗑 General waste.","action":"Dispose safely"}
}

# -------------------------------
# IMAGE UPLOAD
# -------------------------------
uploaded = st.file_uploader("Upload waste image", type=["jpg","png","jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img = img.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    class_index = np.argmax(pred)
    predicted_class = class_names[class_index]
    confidence = np.max(pred)*100

    rule = waste_rules[predicted_class]

    # -------------------------------
    # DISPLAY RESULTS
    # -------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.image(img, caption="Uploaded Waste Image", width=300)
    st.write(f"### 🧠 Predicted Waste: **{predicted_class.upper()}**")
    st.write(f"📊 Confidence: **{confidence:.2f}%**")
    st.write(f"⚙ Type: **{rule['type']}**")
    st.write(rule["message"])
    st.success(f"Recommended Action: {rule['action']}")

    st.markdown('</div>', unsafe_allow_html=True)
