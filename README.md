# AI Resume Screening – Machine Learning Project

## 📌 Suggested Repository Name

**ai-resume-screening-ml**

## 📓 Suggested Notebook Name

**ai_resume_screening_classification.ipynb**

---

## 📖 Project Overview

This project focuses on building a **machine learning–based resume screening system** that predicts whether a candidate should be **shortlisted** based on resume-related features. The notebook performs **end-to-end data analysis**, including data loading, exploratory data analysis (EDA), preprocessing, model training, hyperparameter tuning, and evaluation.

The goal is to compare multiple classification algorithms and identify the best-performing model using robust evaluation metrics.

---

## 🧠 Problem Statement

Manual resume screening is time-consuming and subjective. This project aims to automate the process using supervised machine learning techniques to improve efficiency and consistency in candidate shortlisting.

---

## 📂 Dataset

* **File:** `ai_resume_screening.csv`
* **Target Column:** `shortlisted`
* **Features:**

  * Numerical features (experience, scores, etc.)
  * Categorical features (education level, skills, job role, etc.)

The dataset is analyzed for:

* Data types
* Missing values
* Duplicate records
* Statistical distributions

---

## 🔍 Exploratory Data Analysis (EDA)

The notebook includes comprehensive EDA:

* Dataset shape and schema inspection
* Descriptive statistics (numerical & categorical)
* Missing value and duplicate checks
* Distribution plots (histograms with KDE)
* Value counts for categorical variables
* Pair plots with class-wise separation
* Correlation heatmap for numerical features

These steps help understand feature behavior and relationships with the target variable.

---

## 🛠️ Data Preprocessing

* Train–test split with stratification
* Separation of numerical and categorical features
* **Numerical preprocessing:** StandardScaler
* **Categorical preprocessing:** OrdinalEncoder
* **ColumnTransformer** used for clean and scalable preprocessing

---

## 🤖 Machine Learning Models Used

The following classification models are trained and evaluated:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Classifier (SVC)
* Decision Tree Classifier
* Random Forest Classifier
* Gradient Boosting Classifier

Each model is wrapped in a **Pipeline** and tuned using **GridSearchCV**.

---

## 📊 Model Evaluation

Models are evaluated using multiple metrics:

* Accuracy
* Precision
* Recall
* F1-score

A final comparison table is created and sorted by **F1-score** to identify the best-performing model.

---

## 🏆 Results

* All models are compared under the same preprocessing and evaluation framework
* The best estimator is selected based on validation performance
* Results are summarized in a structured DataFrame for easy comparison

---

## 🧰 Libraries & Tools

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn

---

## 🚀 How to Run

1. Clone the repository
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:

   ```bash
   jupyter notebook ai_resume_screening_classification.ipynb
   ```
4. Run all cells sequentially

---

## 📌 Future Improvements

* Use NLP techniques on raw resume text
* Try advanced models (XGBoost, LightGBM)
* Handle class imbalance (SMOTE, class weights)
* Deploy as a web application (Flask / Streamlit)

---

## 👤 Author

Devendra Kushwah

---

⭐ If you find this project useful, consider giving the repository a star!
