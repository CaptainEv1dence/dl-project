from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from app.inference import load_emotion_model, predict_emotion, top_k


st.set_page_config(
    page_title="FER-2013 Emotion Recognition",
    page_icon="🙂",
    layout="wide",
)

st.title("FER-2013 Emotion Recognition Dashboard")
st.caption("Upload a face image and run emotion classification with a trained model.")

with st.sidebar:
    st.header("Model settings")

    checkpoint_path = st.text_input(
        "Checkpoint path",
        value="outputs/checkpoints/best_efficientnet_b2.pt",
    )

    model_name = st.selectbox(
        "Model",
        options=["efficientnet_b2", "resnet18", "se_cnn", "baseline_cnn"],
        index=0,
    )

    default_size = 224 if model_name == "efficientnet_b2" else 48
    image_size = st.number_input(
        "Image size",
        min_value=48,
        max_value=384,
        value=default_size,
        step=16,
    )

    imagenet_norm = st.checkbox(
        "Use ImageNet normalization",
        value=model_name == "efficientnet_b2",
    )

    device = st.selectbox("Device", options=["auto", "cuda", "cpu"], index=0)

uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["png", "jpg", "jpeg", "webp"],
)

if not Path(checkpoint_path).exists():
    st.warning(f"Checkpoint not found: `{checkpoint_path}`. Train the model first or update the path.")

if uploaded_file is None:
    st.info("Upload an image to run inference.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")

left, right = st.columns([1, 1])

with left:
    st.subheader("Input image")
    st.image(image, use_container_width=True)

try:
    model, torch_device = load_emotion_model(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        device=device,
    )
    result = predict_emotion(
        model=model,
        device=torch_device,
        image=image,
        image_size=int(image_size),
        imagenet_norm=imagenet_norm,
    )
except Exception as exc:
    st.error(f"Inference failed: {exc}")
    st.stop()

with right:
    st.subheader("Prediction")
    st.metric("Emotion", result["label"])
    st.metric("Confidence", f"{result['confidence']:.2%}")
    st.caption(f"Device: `{torch_device}`")

    st.write("Top predictions:")
    for label, prob in top_k(result["probabilities"], k=3):
        st.write(f"- **{label}**: {prob:.2%}")

prob_df = pd.DataFrame(
    {
        "emotion": list(result["probabilities"].keys()),
        "probability": list(result["probabilities"].values()),
    }
).sort_values("probability", ascending=False)

st.subheader("Probability distribution")
st.bar_chart(prob_df, x="emotion", y="probability", use_container_width=True)

with st.expander("Raw probabilities"):
    st.dataframe(prob_df, use_container_width=True, hide_index=True)
