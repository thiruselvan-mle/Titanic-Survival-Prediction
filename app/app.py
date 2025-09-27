import sys
import os
import streamlit as st
import pandas as pd
import base64

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import load_model

# -----------------------------
# Load trained model
# -----------------------------
model_path = os.path.join("models", "random_forest.pkl")
model = load_model(model_path)

# -----------------------------
# Function to set background image
# -----------------------------
def set_background(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background (use your actual path to the image)
set_background(os.path.join("app", "titanic.png"))

# -----------------------------
# Global CSS Styling
# -----------------------------
st.markdown(
    """
    <style>
    /* App title and headings */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;  
        font-weight: 700;
    }

    /* General text */
    .stApp, .stMarkdown, .css-1v0mbdj, .css-16idsys p {
        color: #f0f0f0 !important;  
        font-weight: 500;
        font-size: 16px;
    }

    /* Labels for inputs */
    label, .stSelectbox label, .stNumberInput label {
        color: #e6e6e6 !important;
        font-size: 15px;
        font-weight: 600;
    }

    /* Input boxes */
    .stSelectbox, .stNumberInput input {
        background-color: rgba(255,255,255,0.1) !important;
        color: balck !important;
        border-radius: 8px !important;
    }

    /* Overlay background for main content */
    .block-container {
        background-color: rgba(0, 0, 0, 0.55);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }

    /* Predict Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #ffffff, #cccccc);
        color: #000000;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.7em 1.5em;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.4);
    }

    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #000000, #333333);
        color: #ffffff;
        transform: scale(1.05);
        box-shadow: 0px 6px 15px rgba(0,0,0,0.6);
    }

    /* Footer */
    .stCaption {
        color: #cccccc !important;
        font-size: 14px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# App title
# -----------------------------
st.title("🚢 Titanic Survival Prediction")
st.write("Predict whether a passenger would have survived the Titanic disaster based on their details.")

# -----------------------------
# Layout: two columns for inputs
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
    age = st.number_input("Age", min_value=0, max_value=80, value=25)
    sibsp = st.number_input("Siblings/Spouses aboard (SibSp)", min_value=0, max_value=8, value=0)
    parch = st.number_input("Parents/Children aboard (Parch)", min_value=0, max_value=6, value=0)

with col2:
    sex = st.selectbox("Sex", ["male", "female"])
    fare = st.number_input("Fare", min_value=0.0, max_value=513.3, value=32.2, step=0.1)
    embarked = st.selectbox("Port of Embarkation", ["S (Southampton)", "C (Cherbourg)", "Q (Queenstown)"])

# -----------------------------
# Encode categorical features
# -----------------------------
sex_map = {"male": 0, "female": 1}
embarked_map = {"S (Southampton)": 0, "C (Cherbourg)": 1, "Q (Queenstown)": 2}

sex_encoded = sex_map[sex]
embarked_encoded = embarked_map[embarked]

# -----------------------------
# Feature engineering
# -----------------------------
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

# -----------------------------
# Create DataFrame for prediction
# -----------------------------
input_data = pd.DataFrame([[pclass, sex_encoded, age, sibsp, parch, fare,
                            embarked_encoded, family_size, is_alone]],
                          columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                                   'Embarked', 'FamilySize', 'IsAlone'])

# -----------------------------
# Prediction button
# -----------------------------
if st.button("🔮 Predict Survival"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]  # survival probability

    if prediction == 1:
        result_text = "✅ Survived"
        card_color = "rgba(0,128,0,0.7)"  
        bar_color = "linear-gradient(90deg, #00FF00, #008000)"
    else:
        result_text = "❌ Did not survive"
        card_color = "rgba(255,0,0,0.7)"  
        bar_color = "linear-gradient(90deg, #FF0000, #800000)"

    # Display result card
    st.markdown(f"""
        <div style='background-color:{card_color};
                    padding:15px;
                    border-radius:10px;
                    text-align:center;
                    font-weight:bold;
                    font-size:22px;
                    color:white;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
                    margin-bottom:10px;'>
            {result_text}
        </div>
    """, unsafe_allow_html=True)

    # Probability bar with glow effect
    st.markdown(f"""
        <div style='background-color:#333; border-radius:10px; width:100%; padding:2px;'>
            <div style='width:{int(proba*100)}%;
                        background:{bar_color};
                        padding:6px 0;
                        text-align:center;
                        color:white;
                        font-weight:bold;
                        border-radius:10px;
                        box-shadow: 0 0 15px {bar_color};'>
                {int(proba*100)}%
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.write(f"Probability of survival: **{proba:.2%}**")

# -----------------------------
# Footer
# -----------------------------
st.markdown('<hr style="border: 1px solid white; margin-top:2rem; margin-bottom:1rem;">', unsafe_allow_html=True)
st.caption("Built with ❤️ by Thiruselvan M | Powered by Streamlit & Scikit-Learn")
