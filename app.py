"""
streamlit app for image captioning.
upload a photo, get a description.
"""

import streamlit as st
from PIL import Image
from core import ImageCaptioner, CaptionConfig

st.set_page_config(page_title="Image Captioner", layout="centered")


@st.cache_resource
def load_model():
    cap = ImageCaptioner(CaptionConfig())
    cap.load()
    return cap


st.title("Image Captioning")
st.caption("ViT encoder + GPT-2 decoder")

captioner = load_model()

uploaded = st.file_uploader("upload an image", type=["jpg", "jpeg", "png", "webp"])
num_captions = st.slider("number of captions", 1, 5, 3)

if uploaded:
    image = Image.open(uploaded)
    st.image(image, use_container_width=True)

    if st.button("generate captions", type="primary", use_container_width=True):
        with st.spinner("running inference..."):
            captions, time_ms = captioner.caption(image, num_captions=num_captions)

        st.divider()
        for i, cap in enumerate(captions):
            if i == 0:
                st.markdown(f"**best:** {cap}")
            else:
                st.write(f"{i+1}. {cap}")

        st.caption(f"generated in {time_ms:.0f}ms on {captioner.health()['device']}")
