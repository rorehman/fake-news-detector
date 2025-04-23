# 📰 Fake News Detector 🔍

## 🧠 Objective  
Build a machine learning model that can classify news articles as either **"Fake"** or **"Real"** based solely on their text content. The final goal is to deploy this model as a web app to help users identify potential misinformation online.

---

## 🔍 Problem Type  
This is a **binary classification** problem. Given a news article’s content, the model predicts whether the article is likely fake or real.

---

## ✅ Success Criteria  
- Achieve **≥85% accuracy** on the test set  
- Maintain **balanced precision and recall** to prevent class bias  
- Build a clean, modular **Scikit-learn pipeline**  
- Deploy via a lightweight and interactive **web app**

---

## 📦 Dataset  
We are using a publicly available dataset containing thousands of labeled **fake** and **real** news articles. The data will be cleaned, analyzed, and split into training, validation, and test sets.

### Directory Structure:
data/ ├── raw/ │ ├── Fake.csv │ └── True.csv └── processed/

---

## 🔧 ML Workflow  
- Exploratory Data Analysis (EDA)  
- Feature Engineering (e.g., word count, TF-IDF)  
- Model Training & Evaluation (e.g., Logistic Regression, Naive Bayes)  
- Hyperparameter Tuning  
- Finalize a robust ML Pipeline using Scikit-learn  
- Deployment via Streamlit (or Flask)

---

## 🛠 Tech Stack  
- Python 3.10  
- Jupyter Notebooks (for prototyping & EDA)  
- Scikit-learn  
- Pandas / NumPy / Matplotlib / Seaborn  
- NLP tools (NLTK or spaCy)  
- Streamlit or Flask (for web deployment)  
- Git & GitHub (for version control)  
- Docker *(optional for deployment)*

---

## 🚧 Status  
📅 Currently in Week 1: Exploratory Data Analysis  
✅ Dataset loaded  
🔍 Investigating word counts, text length, class balance

