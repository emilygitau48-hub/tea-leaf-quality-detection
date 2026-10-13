from pathlib import Path

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "tea_leaf_model.keras"

st.set_page_config(
    page_title="Tea Leaf Quality Detection",
    page_icon="🍃",
    layout="centered"
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


st.title("🍃 Tea Leaf Quality Detection System")

st.write(
    "Upload a photograph of an individual tea leaf "
    "to receive an experimental GOOD or BAD classification."
)

st.info(
    "For best results in this prototype, photograph one leaf "
    "at a time under clear lighting."
)

if not MODEL_PATH.exists():
    st.error("The model file tea_leaf_model.keras is missing.")
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"The model could not be loaded: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a tea leaf photograph",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Selected tea leaf photograph",
            use_container_width=True
        )

        if st.button("Detect Tea Leaf Quality", type="primary"):
            with st.spinner("Analysing the photograph..."):
                resized_image = image.resize((180, 180))
                image_array = np.array(resized_image, dtype=np.float32)
                image_array = image_array / 255.0
                image_array = np.expand_dims(image_array, axis=0)

                prediction = float(
                    model.predict(image_array, verbose=0)[0][0]
                )

            if prediction >= 0.5:
                st.success("Predicted class: GOOD tea leaf")
            else:
                st.warning("Predicted class: BAD tea leaf")

            st.caption(
                f"Model output score: {prediction:.4f}"
            )

            st.warning(
                "This is an experimental prediction, not a confirmed "
                "assessment of tea quality. The model was trained on "
                "a small dataset and may make mistakes."
            )

    except Exception as error:
        st.error(f"Could not process this image: {error}")
else:
    st.caption("Choose an image above to begin.")