# Titanic Survival Prediction 🚢✨

Predict whether a passenger survived the Titanic disaster using **Python**, **Pandas**, **Scikit-Learn**, and **Streamlit**.  

- Interactive web app with **custom UI**  
- Predicts **survival probability** based on passenger details (class, age, gender, fare)  
- Uses **Random Forest Classifier**  
- Beginner-friendly project to learn **data analysis, feature engineering, ML, and deployment**  

📊 Demo:  
<img src="app/titanic.png" alt="APP DEMO" width="100%" height="600">

📚 Dataset: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)  

▶️ Run the app:  
```bash
git clone https://github.com/your-username/titanic-survival.git
cd titanic-survival
pip install -r requirements.txt
streamlit run app/app.py
Author: Thiruselvan M | Built with ❤️ using Streamlit & Scikit-Learn

📌 Project Overview
This project predicts whether a passenger would have survived the Titanic disaster based on personal and travel details such as class, age, gender, fare, and more.
It uses a Random Forest Classifier trained on the Titanic dataset.

This repository is designed for beginners starting their Machine Learning journey, showing how to approach a Kaggle competition step by step in a simple and clear way.

📂 Project Structure
bash
Copy code
titanic-survival/
├─ data/          # Dataset files
├─ notebooks/     # Jupyter notebooks for EDA & model training
├─ src/           # Source code (model loading, utils, preprocessing)
├─ models/        # Saved ML models (.pkl files)
├─ app/           # Streamlit app files (UI + background image)
├─ requirements.txt  # Python dependencies
├─ README.md      # Project documentation
└─ .gitignore     # Ignored files
⚡ Quick Start
bash
Copy code
# Clone the repo
git clone https://github.com/your-username/titanic-survival.git
cd titanic-survival

# Create & activate virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/app.py
Open http://localhost:8501 in your browser.

🛠️ Features
Interactive web app with custom background and styled UI

Predicts survival probability of passengers

Encodes categorical features (Sex, Embarked)

Includes feature engineering: family size & is_alone

Model trained with Random Forest Classifier

🔹 Learning Goals
This project teaches beginners:

Data Handling & Cleaning with Pandas

Exploratory Data Analysis (EDA) with Matplotlib/Seaborn

Feature Engineering

Applying Supervised Machine Learning (Random Forest Classifier)

Evaluating models using accuracy and cross-validation

Deploying an interactive Streamlit Web App

🤝 Contributing
Contributions are welcome!
Feel free to fork this repo, create a branch, and submit a pull request.

📜 License
This project is licensed under the MIT License – see the LICENSE file for details.

👨‍💻 Author
Thiruselvan M
Built with ❤️ using Streamlit & Scikit-Learn
