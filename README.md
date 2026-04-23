# AI-Powered Student Risk Prediction System

This project is a final year computer science project that predicts whether a secondary school student is at risk of failing their final mathematics examination. It uses machine learning on the UCI Student Performance Dataset and includes a Streamlit-based prototype interface designed as a decision-support tool for educators.

## Features
- Binary pass/fail risk prediction
- Logistic Regression, Random Forest, and Gradient Boosting model comparison
- Stratified 5-fold cross-validation
- Threshold tuning using training data only
- Streamlit prototype interface
- Global feature importance visualisation
- Dynamic educator recommendations
- Borderline profile warning overlay

## Files
- `model.py` — training, preprocessing, evaluation, threshold tuning, and model saving
- `app.py` — Streamlit prototype for risk prediction
- `requirements.txt` — Python dependencies

## Setup
1. Install Python 3.10 or above.
2. Install the required packages:
   `pip install -r requirements.txt`

## Run the project
1. Make sure the dataset file is in the correct folder.
2. Run the training pipeline:
   `python model.py`
3. Launch the Streamlit app:
   `python -m streamlit run app.py`

## Dataset
This project uses the UCI Student Performance Dataset. If the dataset is not included in this repository, download it from the UCI Machine Learning Repository and place `student-mat.csv` in the correct project folder before running `model.py`.

## Author
Amr Nasr