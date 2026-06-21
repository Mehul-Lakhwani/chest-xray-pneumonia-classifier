import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# ── Page config ──
st.set_page_config(
    page_title="Chest X-Ray Pneumonia Classifier",
    page_icon="🫁",
    layout="centered"
)

# ── Load model ──
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("pneumonia_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── UI ──
st.title("🫁 Chest X-Ray Pneumonia Classifier")
st.markdown("""
**Built with PyTorch + ResNet18 Transfer Learning**  
Upload a chest X-ray image to detect whether it shows signs of **Pneumonia** or is **Normal**.

> Model accuracy: **89.58%** on 624 test images  
> Architecture: ResNet18 pretrained on ImageNet, fine-tuned on 5,216 chest X-rays
""")

st.divider()

uploaded_file = st.file_uploader("Upload a Chest X-Ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-Ray", use_column_width=True)

    with st.spinner("Analyzing..."):
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            normal_prob = float(probs[0])
            pneumonia_prob = float(probs[1])

    st.divider()
    st.subheader("Prediction")

    if pneumonia_prob > normal_prob:
        st.error(f"🔴 PNEUMONIA detected — {pneumonia_prob*100:.1f}% confidence")
    else:
        st.success(f"🟢 NORMAL — {normal_prob*100:.1f}% confidence")

    st.subheader("Confidence Scores")
    st.progress(normal_prob, text=f"Normal: {normal_prob*100:.1f}%")
    st.progress(pneumonia_prob, text=f"Pneumonia: {pneumonia_prob*100:.1f}%")

    st.divider()
    st.caption("⚠️ This tool is for educational purposes only and is not a medical diagnostic device.")
