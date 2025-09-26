import sys 
import os 
import streamlit as st
import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import load_model


model = load_model(r"D:\Thiru\ML_Projects\Titanic-Survival-Prediction\models\random_forest.pkl")

st.title("Titanic Survival Prediction")
st.write("Enter passenger details to predict survival:")


pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Number of Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=32.2)
embarked = st.selectbox("Port of Embarkation (Embarked)", ["S", "C", "Q"])


sex_map = {"male": 0, "female": 1}
embarked_map = {"S": 0, "C": 1, "Q": 2}

sex_encoded = sex_map[sex]
embarked_encoded = embarked_map[embarked]


family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

# Create DataFrame
input_data = pd.DataFrame([[
    pclass, sex_encoded, age, sibsp, parch, fare,
    embarked_encoded, family_size, is_alone
]], columns=[
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
    'Embarked', 'FamilySize', 'IsAlone'
])

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    survival = "Survived ✅" if prediction == 1 else "Did not survive ❌"
    st.success(f"Prediction: {survival}")
