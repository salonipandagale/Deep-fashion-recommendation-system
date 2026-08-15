import streamlit as st
import numpy as np
import pickle
import tempfile
from tensorflow.keras.models import load_model
from PIL import Image
from fashion_utils import preprocess_image, extract_features, recommend

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Fashion Recommender",
    page_icon="👗",
    layout="wide"
)
import base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


background_image = get_base64_image("background.jpg")
# ---------------- CSS ----------------
st.markdown(
    f"""
    <style>

    header {{
        visibility: hidden;
    }}

    footer {{
        visibility: hidden;
    }}

    #MainMenu {{
        visibility: hidden;
    }}

    .stApp {{
        background:
            linear-gradient(
                rgba(0,0,0,0.5),
                rgba(0,0,0,0.6)
            ),
            url("data:image/jpeg;base64,{background_image}");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .title {{
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        color: white;
    }}

    .subtitle {{
        text-align: center;
        font-size: 18px;
        color: #eee;
        margin-bottom: 30px;
    }}

    h3 {{
        color: #f5f5f5 !important;
        font-weight: 600;
        text-shadow: 0px 2px 6px rgba(0,0,0,0.6);
    }}

    h2 {{
        color: #f5f5f5 !important;
        font-weight: 600;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------- TITLE ----------------

st.markdown('<div class="title">👗 AI Fashion Recommender</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload your outfit & discover similar styles instantly</div>',
    unsafe_allow_html=True
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model_and_data():
    model = load_model("vgg16_feature_extractor.h5")
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
        image_bytes = uploaded_file.getvalue()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as tmp:
            tmp.write(image_bytes)
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