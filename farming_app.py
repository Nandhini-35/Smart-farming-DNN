import streamlit as st
import numpy as np
import pickle
import time
from sklearn.metrics import accuracy_score

# -----------------------------
# Load Model
# -----------------------------
model = pickle.load(open("smart_farming_model.pkl", "rb"))

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="🌾 Smart Farming AI Assistant", page_icon="🌿", layout="wide")

# -----------------------------
# CSS STYLING
# -----------------------------
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            background-attachment: fixed;
        }
        .main {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            padding: 40px;
            margin-top: 50px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        h1 {
            color: #fff;
            text-align: center;
            font-family: 'Poppins', sans-serif;
            font-size: 2.8rem;
            text-shadow: 0 0 10px #00FF88;
            animation: glow 2s ease-in-out infinite alternate;
        }
        @keyframes glow {
            from { text-shadow: 0 0 10px #00FF88; }
            to { text-shadow: 0 0 30px #00ffbb, 0 0 50px #00ffbb; }
        }
        label {
            color: #f2f2f2 !important;
            font-weight: 600;
        }
        .stButton button {
            background: linear-gradient(45deg, #00ff88, #00bfa5);
            border: none;
            color: white;
            font-weight: 700;
            border-radius: 12px;
            padding: 10px 24px;
            transition: 0.3s;
        }
        .stButton button:hover {
            transform: scale(1.05);
            background: linear-gradient(45deg, #00e676, #00c853);
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.markdown("<h1>🌾 Smart Farming AI Assistant 🌾</h1>", unsafe_allow_html=True)
st.markdown('<div class="main">', unsafe_allow_html=True)

# -----------------------------
# Slider Inputs
# -----------------------------
st.subheader("🧪 Adjust Soil and Weather Parameters")

nitrogen = st.slider("Nitrogen (N)", 0.0, 150.0, 50.0)
phosphorus = st.slider("Phosphorus (P)", 0.0, 150.0, 50.0)
potassium = st.slider("Potassium (K)", 0.0, 150.0, 50.0)
temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0)
ph = st.slider("pH Value", 0.0, 14.0, 6.5)
rainfall = st.slider("Rainfall (mm)", 0.0, 500.0, 200.0)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🌱 Predict Best Crop"):
    with st.spinner("Analyzing with AI... 🌿"):
        time.sleep(2)
        input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)
        result = prediction[0]

    st.success(f"✅ Recommended Crop: **{result}**")

# -----------------------------
# Display Model Accuracy
# -----------------------------
try:
    from sklearn.model_selection import train_test_split
    import pandas as pd

    df = pd.read_csv("Crop_recommendation.csv")
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100

    st.markdown(f"### 📊 Model Accuracy: **{acc:.2f}%**")
except Exception as e:
    st.warning("⚠️ Accuracy can't be calculated (dataset missing locally).")

st.markdown("</div>", unsafe_allow_html=True)
