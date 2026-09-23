
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="AI Expected Goals (xG) Predictor", layout="centered")
st.title("⚽ AI Expected Goals (xG) Predictor")
st.write("Evaluate the probability of any football shot using physics and machine learning.")

# Load the saved model package
data = joblib.load('xg_model_package.pkl')
model = data['model']
model_features = data['features']

# UI Inputs
st.header("Shot Characteristics")
col1, col2 = st.columns(2)

with col1:
    x_coord = st.slider("Shot X Coordinate (Distance from own goal line)", 60.0, 119.5, 105.0, 0.5)
    y_coord = st.slider("Shot Y Coordinate (Left to Right width)", 0.0, 80.0, 40.0, 0.5)
    under_pressure = st.selectbox("Is the shooter under defender pressure?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

with col2:
    body_part = st.selectbox("Shooting Body Part", ["Right Foot", "Left Foot", "Head", "Other"])
    technique = st.selectbox("Shot Technique", ["Normal", "Volley", "Half-Volley", "Lob", "Diving Header"])
    pattern = st.selectbox("Play Pattern", ["Regular Play", "From Counter", "From Corner", "From Free Kick", "Other"])

# Calculate geometry features on the fly
dist = np.sqrt((120 - x_coord)**2 + (40 - y_coord)**2)

v1 = np.array([120 - x_coord, 36 - y_coord])
v2 = np.array([120 - x_coord, 44 - y_coord])
norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
if norm_v1 == 0 or norm_v2 == 0:
    angle = 0.0
else:
    cos_theta = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))

# Build raw input dict
input_data = {
    'distance': dist,
    'angle': angle,
    'under_pressure': under_pressure,
    'shot_body_part': body_part,
    'shot_technique': technique,
    'play_pattern': pattern
}

# Transform to match training columns (One-Hot Encoding alignment)
input_df = pd.DataFrame([input_data])
encoded_df = pd.get_dummies(input_df)

# Create an empty template row matching training schema
final_features = pd.DataFrame(0, index=[0], columns=model_features)
for col in encoded_df.columns:
    if col in final_features.columns:
        final_features[col] = encoded_df[col].values

# Explicitly ensure numerical continuous inputs are assigned properly
final_features['distance'] = dist
final_features['angle'] = angle
final_features['under_pressure'] = under_pressure

# Make Prediction
if st.button("Predict xG Probability", type="primary"):
    probability = model.predict_proba(final_features)[0, 1]
    
    st.metric(label="Calculated xG Value", value=f"{probability:.2%}")
    st.caption(f"Calculated Metrics -> Distance: {dist:.1f} yards | Visual Angle: {angle:.1f}°")
    
    if probability > 0.40:
        st.success("🔥 This is a high-grade scoring opportunity!")
    elif probability > 0.15:
        st.warning("⚠️ Decent chance, requires strong execution.")
    else:
        st.error("📉 Low probability shot location/context.")
