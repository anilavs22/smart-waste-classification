import streamlit as st
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Smart Waste Classification System",
    layout="wide"
)

# =====================================================
# TITLE
# =====================================================

st.title("♻️ Smart Waste Classification System")

st.write(
    "Deep Learning-based Waste Classification using TensorFlow and PyTorch"
)

# =====================================================
# WASTE CLASSES
# =====================================================

classes = [
    'battery',
    'biological',
    'cardboard',
    'clothes',
    'glass',
    'metal',
    'paper',
    'plastic',
    'shoes',
    'trash'
]

# =====================================================
# WASTE INFORMATION
# =====================================================

waste_info = {
    "battery": "⚠️ Hazardous electronic waste.",
    "biological": "🌱 Organic biodegradable waste.",
    "cardboard": "📦 Recyclable cardboard material.",
    "clothes": "👕 Reusable textile waste.",
    "glass": "🍾 Recyclable glass material.",
    "metal": "🔩 Recyclable metal waste.",
    "paper": "📄 Recyclable paper waste.",
    "plastic": "♻️ Recyclable plastic material.",
    "shoes": "👟 Reusable footwear waste.",
    "trash": "🗑️ Non-recyclable general waste."
}

# =====================================================
# LOAD TENSORFLOW MODEL
# =====================================================

tf_model = tf.keras.models.load_model(
    "models/waste_model.h5",
    compile=False
)

# =====================================================
# LOAD PYTORCH MODEL
# =====================================================

device = torch.device("cpu")

torch_model = models.resnet18(weights=None)

torch_model.fc = nn.Linear(
    torch_model.fc.in_features,
    10
)

torch_model.load_state_dict(
    torch.load(
        "models/waste_pytorch.pth",
        map_location=device
    )
)

torch_model.eval()

# =====================================================
# IMAGE TRANSFORM
# =====================================================

torch_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_image(image):

    # =============================================
    # TensorFlow Prediction
    # =============================================

    tf_img = image.resize((224,224))

    tf_array = np.array(tf_img) / 255.0

    tf_array = np.expand_dims(tf_array, axis=0)

    tf_prediction = tf_model.predict(
        tf_array,
        verbose=0
    )

    tf_class = classes[np.argmax(tf_prediction)]

    tf_confidence = float(np.max(tf_prediction))

    # =============================================
    # PyTorch Prediction
    # =============================================

    torch_img = torch_transform(image)

    torch_img = torch_img.unsqueeze(0)

    with torch.no_grad():

        outputs = torch_model(torch_img)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        torch_confidence, predicted = torch.max(
            probabilities,
            1
        )

    torch_class = classes[predicted.item()]

    torch_confidence = torch_confidence.item()

    return (
        tf_class,
        tf_confidence,
        torch_class,
        torch_confidence,
        tf_prediction[0],
        probabilities.numpy()[0]
    )

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📌 About Project")

st.sidebar.info(
    """
Smart Waste Classification System
using TensorFlow and PyTorch.

Features:
- Deep Learning Classification
- TensorFlow Prediction
- PyTorch Prediction
- Confidence Visualization
- Waste Category Insights
"""
)

# =====================================================
# IMAGE UPLOAD SECTION
# =====================================================

st.header("📂 Upload Waste Image")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# MAIN APP
# =====================================================

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1])

    # =============================================
    # DISPLAY IMAGE
    # =============================================

    with col1:

        st.image(
            image,
            caption="Uploaded Waste Image",
            width=350
        )

    # =============================================
    # GET PREDICTIONS
    # =============================================

    (
        tf_class,
        tf_confidence,
        torch_class,
        torch_confidence,
        tf_probs,
        torch_probs
    ) = predict_image(image)

    # =============================================
    # DISPLAY RESULTS
    # =============================================

    with col2:

        st.subheader("🔍 Prediction Results")

        st.metric(
            label="TensorFlow Prediction",
            value=tf_class,
            delta=f"{tf_confidence:.2f}"
        )

        st.metric(
            label="PyTorch Prediction",
            value=torch_class,
            delta=f"{torch_confidence:.2f}"
        )

        st.info(
            waste_info[tf_class]
        )

    # =============================================
    # COMPARISON TABLE
    # =============================================

    st.subheader("📊 Model Comparison")

    comparison_df = pd.DataFrame({

        "Model": [
            "TensorFlow",
            "PyTorch"
        ],

        "Prediction": [
            tf_class,
            torch_class
        ],

        "Confidence": [
            round(tf_confidence, 2),
            round(torch_confidence, 2)
        ]
    })

    st.table(comparison_df)

    # =============================================
    # PROBABILITY CHARTS
    # =============================================

    col3, col4 = st.columns(2)

    # =============================================
    # TensorFlow Chart
    # =============================================

    with col3:

        st.subheader("📈 TensorFlow Probabilities")

        tf_chart = pd.DataFrame({

            "Class": classes,
            "Probability": tf_probs

        })

        st.bar_chart(
            tf_chart.set_index("Class")
        )

    # =============================================
    # PyTorch Chart
    # =============================================

    with col4:

        st.subheader("🔥 PyTorch Probabilities")

        torch_chart = pd.DataFrame({

            "Class": classes,
            "Probability": torch_probs

        })

        st.bar_chart(
            torch_chart.set_index("Class")
        )

    # =============================================
    # TOP 3 PREDICTIONS
    # =============================================

    col5, col6 = st.columns(2)

    # =============================================
    # TensorFlow Top 3
    # =============================================

    with col5:

        st.subheader("🏆 Top 3 TensorFlow Predictions")

        tf_top3_idx = np.argsort(
            tf_probs
        )[-3:][::-1]

        for idx in tf_top3_idx:

            st.write(
                f"{classes[idx]} → {tf_probs[idx]:.2f}"
            )

    # =============================================
    # PyTorch Top 3
    # =============================================

    with col6:

        st.subheader("🔥 Top 3 PyTorch Predictions")

        torch_top3_idx = np.argsort(
            torch_probs
        )[-3:][::-1]

        for idx in torch_top3_idx:

            st.write(
                f"{classes[idx]} → {torch_probs[idx]:.2f}"
            )