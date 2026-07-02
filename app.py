import streamlit as st
import numpy as np
import pickle
import tempfile
from tensorflow.keras.models import load_model
from PIL import Image
from fashion_utils import preprocess_image, extract_features, recommend

import gdown
import os

MODEL_PATH = "vgg16_feature_extractor.h5"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1dDgmSzk8mWg8K3hIDk2M5a-wZv3nRB0m"
    gdown.download(url, MODEL_PATH, quiet=False)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fashion Recommender",
    page_icon="👗",
    layout="wide"
)

# ---------------- CSS ----------------

st.markdown("""
<style>

/* REMOVE HEADER */
header {visibility: hidden;}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}

/* Background Image */
.stApp {
    background: 
        linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.6)),
        url("https://fashionsuggest.in/wp-content/uploads/Celebrity-outfit_feature_image-1021x580.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Make text visible on dark bg */
.title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: white;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #eee;
    margin-bottom: 30px;
}


/* ONLY these headings */
h3 {
    color: #f5f5f5 !important;   /* soft white */
    font-weight: 600;
}
h2 {
    color: #f5f5f5 !important;   /* soft white */
    font-weight: 600;
}


/* Optional: make them slightly glow for visibility */
h3 {
    text-shadow: 0px 2px 6px rgba(0,0,0,0.6);
}
</style>
""", unsafe_allow_html=True)
# ---------------- TITLE ----------------

st.markdown('<div class="title">👗 AI Fashion Recommender</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload your outfit & discover similar styles instantly</div>',
    unsafe_allow_html=True
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model_and_data():
    model = load_model("model.keras", compile=False)
    features = pickle.load(open("features.pkl", "rb"))
    filenames = pickle.load(open("filenames.pkl", "rb"))
    return model, features, filenames

model, features, filenames = load_model_and_data()

# ---------------- MAIN CARD ----------------

st.markdown('<div class="card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📤 Upload a fashion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 🖼️ Preview")
        st.image(uploaded_file, use_container_width=True)

    with col2:
        st.markdown("### 🔍 Find Similar Styles")
        find_btn = st.button("✨ Generate Recommendations")

    if find_btn:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        with st.spinner("Analyzing fashion style... 👗"):
            results = recommend(temp_path, model, features, filenames)

        st.markdown("---")
        st.markdown('<h2 class="section-title">🎯 Recommended For You</h2>', unsafe_allow_html=True)

        cols = st.columns(5)
        for i, col in enumerate(cols):
            if i < len(results):
                col.image(results[i], use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)
