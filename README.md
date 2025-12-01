# 🧬 GenoMatch – Disease Inheritance Predictor

GenoMatch is a deep-learning based application that predicts the likelihood of inheriting disease risks using health, lifestyle, and family-history data. Upload a CSV file and instantly receive YES/NO predictions along with risk percentages through a modern Streamlit interface.

---

## 🚀 Features
- Deep Learning (DNN) model trained on medical-style dataset
- Predicts YES/NO inheritance risk + probability score
- CSV upload for batch predictions
- Automatic encoding & scaling
- Styled Streamlit UI with custom CSS
- Fast, lightweight, and easy to run

---

## 🧠 Tech Stack
- Python
- TensorFlow / Keras
- Pandas, NumPy, Scikit-learn
- Streamlit
- Joblib

---

## 📁 Project Structure
GENOMATCH/
│── app.py
│── train_model.py
│── disease_risk_model.keras
│── artifacts.pkl
│── sample_test.csv
│── README.md

---

## 📦 Installation
pip install -r requirements.txt

---

## ▶️ Run the App
streamlit run app.py

Your app will open at:
http://localhost:8501

---

## 📤 How to Use
1. Upload a CSV file  
2. The app preprocesses everything automatically  
3. Get predictions:
   - YES / NO  
   - Risk percentage (%)  
4. View overall summary at the bottom  

---

## 🧪 Sample Input (CSV)
age,bmi,smoking,alcohol,exercise,gender,family_history
45,27.5,1,0,1,M,1
32,22.1,0,1,2,F,0

---

## 📈 Example Output
age | bmi  | inherit_risk_percent | inherit_prediction
----|------|-----------------------|---------------------
45  | 27.5 | 72.14%                | YES
32  | 22.1 | 18.55%                | NO

---

## 💡 Why GenoMatch?
GenoMatch demonstrates how AI can support preventive healthcare by estimating inherited disease tendencies using deep learning. It showcases complete AIML workflow: preprocessing, training, deployment, and UI development — making it a strong portfolio project.

---

live demo:https://genomatch.streamlit.app/
