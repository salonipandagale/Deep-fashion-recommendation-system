import os
import base64
import pickle
import tempfile

import streamlit as st
from tensorflow.keras.models import load_model

from fashion_utils import recommend


# PAGE CONFIG

st.set_page_config(
    page_title="Fashion Recommender",
    page_icon="👗",
    layout="wide"
)


# HELPER: GET BASE64 BACKGROUND IMAGE

def get_base64_image(image_path):
    """
    Convert an image to Base64 so it can be used
    directly inside CSS.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(
            image_file.read()
        ).decode("utf-8")


# Get project directory
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Background image path
BACKGROUND_PATH = os.path.join(
    BASE_DIR,
    "background.jpg"
)


# Check background image exists
if not os.path.exists(BACKGROUND_PATH):
    st.error(
        f"Background image not found: {BACKGROUND_PATH}"
    )
    st.stop()


background_image = get_base64_image(
    BACKGROUND_PATH
)


# CSS

st.markdown(
    f"""
    <style>

    /* Hide Streamlit header */
    header {{
        visibility: hidden;
    }}

    /* Hide footer */
    footer {{
        visibility: hidden;
    }}

    /* Hide hamburger menu */
    #MainMenu {{
        visibility: hidden;
    }}

    /* Background */
    .stApp {{
        background:
            linear-gradient(
                rgba(0,0,0,0.5),
                rgba(0,0,0,0.6)
            ),
            url(
                "data:image/jpeg;base64,{background_image}"
            );

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    /* Main title */
    .title {{
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        color: white;
        margin-top: 20px;
    }}

    /* Subtitle */
    .subtitle {{
        text-align: center;
        font-size: 18px;
        color: #eeeeee;
        margin-bottom: 30px;
    }}

    /* Section headings */
    h3 {{
        color: #f5f5f5 !important;
        font-weight: 600;
        text-shadow:
            0px 2px 6px rgba(0,0,0,0.6);
    }}

    h2 {{
        color: #f5f5f5 !important;
        font-weight: 600;
        text-shadow:
            0px 2px 6px rgba(0,0,0,0.6);
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# TITLE

st.markdown(
    '<div class="title">👗 AI Fashion Recommender</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload your outfit & discover similar styles instantly'
    '</div>',
    unsafe_allow_html=True
)


# LOAD MODEL + DATA

@st.cache_resource
def load_model_and_data():

    model_path = os.path.join(
        BASE_DIR,
        "vgg16_feature_extractor.h5"
    )

    features_path = os.path.join(
        BASE_DIR,
        "features.pkl"
    )

    filenames_path = os.path.join(
        BASE_DIR,
        "filenames.pkl"
    )

    # Check required files
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"Features file not found: {features_path}"
        )

    if not os.path.exists(filenames_path):
        raise FileNotFoundError(
            f"Filenames file not found: {filenames_path}"
        )

    # Load model
    model = load_model(
        model_path,
        compile=False
    )

    # Load extracted features
    with open(features_path, "rb") as f:
        features = pickle.load(f)

    # Load image filenames
    with open(filenames_path, "rb") as f:
        filenames = pickle.load(f)

    # --------------------------------------------------------
    # IMPORTANT:
    # Convert Windows paths to Linux-compatible paths.
    #
    # Windows:
    # images\image_0004.jpg
    #
    # Linux/Docker:
    # images/image_0004.jpg
    # --------------------------------------------------------

    normalized_filenames = []

    for filename in filenames:

        filename = str(filename)

        # Convert backslashes to forward slashes
        filename = filename.replace("\\", "/")

        # Remove accidental ./ prefix
        if filename.startswith("./"):
            filename = filename[2:]

        # Convert to absolute path
        filename = os.path.join(
            BASE_DIR,
            filename
        )

        # Normalize path
        filename = os.path.normpath(
            filename
        )

        normalized_filenames.append(
            filename
        )

    return (
        model,
        features,
        normalized_filenames
    )


# Load everything
try:

    model, features, filenames = (
        load_model_and_data()
    )

except Exception as e:

    st.error(
        "❌ Error loading the model or image data."
    )

    st.exception(e)

    st.stop()


# MAIN SECTION

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)


# IMAGE UPLOAD

uploaded_file = st.file_uploader(
    "📤 Upload a fashion image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


if uploaded_file is not None:

    col1, col2 = st.columns(
        [1, 2]
    )

    # --------------------------------------------------------
    # PREVIEW
    # --------------------------------------------------------

    with col1:

        st.markdown(
            "### 🖼️ Preview"
        )

        st.image(
            uploaded_file,
            width="stretch"
        )

    # --------------------------------------------------------
    # BUTTON
    # --------------------------------------------------------

    with col2:

        st.markdown(
            "### 🔍 Find Similar Styles"
        )

        find_btn = st.button(
            "✨ Generate Recommendations",
            type="primary"
        )

    # ========================================================
    # GENERATE RECOMMENDATIONS
    # ========================================================

    if find_btn:

        image_bytes = (
            uploaded_file.getvalue()
        )

        temp_path = None

        try:

            # ------------------------------------------------
            # Save uploaded image temporarily
            # ------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg"
            ) as tmp:

                tmp.write(image_bytes)

                temp_path = tmp.name

            # ------------------------------------------------
            # Generate recommendations
            # ------------------------------------------------

            with st.spinner(
                "Analyzing fashion style... 👗"
            ):

                results = recommend(
                    temp_path,
                    model,
                    features,
                    filenames,
                    top_n=5
                )

            # ------------------------------------------------
            # Results heading
            # ------------------------------------------------

            st.markdown("---")

            st.markdown(
                '<h2 class="section-title">'
                '🎯 Recommended For You'
                '</h2>',
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # Display recommendations
            # ------------------------------------------------

            cols = st.columns(5)

            displayed_count = 0

            for i, col in enumerate(cols):

                if i >= len(results):
                    continue

                result_path = results[i]

                # --------------------------------------------
                # Normalize path again
                # --------------------------------------------

                result_path = str(
                    result_path
                ).replace("\\", "/")

                # --------------------------------------------
                # Check image exists
            

                if os.path.exists(
                    result_path
                ):

                    with col:

                        st.image(
                            result_path,
                            width="stretch"
                        )

                        displayed_count += 1

                else:

                    with col:

                        st.error(
                            "Image not found"
                        )

                        st.caption(
                            result_path
                        )

            # No valid results
           

            if displayed_count == 0:

                st.warning(
                    "No recommendation images "
                    "could be displayed."
                )

        except Exception as e:

            st.error(
                "❌ Something went wrong while "
                "generating recommendations."
            )

            st.exception(e)

        finally:

            

            if (
                temp_path is not None
                and os.path.exists(temp_path)
            ):

                try:
                    os.remove(temp_path)
                except Exception:
                    pass



st.markdown(
    "</div>",
    unsafe_allow_html=True
)
