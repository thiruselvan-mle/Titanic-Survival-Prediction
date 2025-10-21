 # Titanic Survival Prediction

An end-to-end Machine Learning project that predicts whether a passenger would have survived the Titanic disaster based on their details such as **class, age, gender, fare, and family information**.  

This project is built to help beginners understand the complete ML pipeline and how to deploy a predictive model using **Streamlit**.

---

<img src=app/titanic1.gif width=100% height=600>

## Project Overview
- Cleaned and preprocessed the Titanic dataset (from [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)).  
- Engineered useful features such as **FamilySize** and **IsAlone**.  
- Trained a **Random Forest Classifier** for survival prediction.  
- Built an interactive web app with **Streamlit** to showcase predictions.  

---

## Project Structure
```bash

project-name/
├─ data/ # Dataset (Raw data not pushed to GitHub)
├─ notebooks/ # Jupyter notebooks (EDA, model training)
├─ src/ # Source code (model.py, utils, etc.)
├─ models/ # Saved ML models (ignored in GitHub)
├─ app/ # Streamlit app (app.py)
├─ requirements.txt # Dependencies
├─ README.md # Project documentation
└─ .gitignore # Files to ignore

```

---
## Installation

 1. Clone the repository:
   ```bash
   git clone https://github.com/thiruselvan-mle/project.git
   cd Titanic-Survival-Prediction

   ```
 2. Create a virtual environment:
   ```bash
   python -m venv venv
   # On Mac/Linux
   source venv/bin/activate
   # On Windows
   venv\Scripts\activate      
   ```
 3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
 4. Run the Streamlit App
   ```bash
   streamlit run app/app.py
   Then open the browser at http://localhost:8501
   ```
---
## Demo
  User inputs passenger details (Age, Sex, Pclass, Fare, Family info).

  Model predicts Survived ✅ or Did Not Survive ❌.

  Probability of survival is also shown with a progress bar.


  <img src=app/demo.png width=100% height=600>

---
## Model Performance   
  Algorithm: Random Forest Classifier

  Accuracy: ~81% (varies depending on training)

  Evaluated using cross-validation and test split.

---
## Features
  Clean ML pipeline (EDA → Preprocessing → Feature Engineering → Modeling).

  Interactive app with Streamlit.

  Modular project structure for scalability.

  Beginner-friendly and easy to extend.

---
## Acknowledgments
  Kaggle Titanic Dataset

  Scikit-learn, Pandas, NumPy, Matplotlib, Streamlit

 ---
## Author
  Thiruselvan Muthuraman
  
  GitHub: @thiruselvan-mle
   
  LinkedIn: https://www.linkedin.com/in/thiruselvan-muthuraman-7506b6387/

## 📜 License
**This project is licensed under the MIT License**