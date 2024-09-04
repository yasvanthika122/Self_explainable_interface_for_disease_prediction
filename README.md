# Self-explainable Interface for Disease Prediction

This project is a Streamlit-based web application for predicting diseases such as diabetes and stroke using machine learning models. The application provides a user-friendly interface where users can input relevant health parameters and receive predictions on the likelihood of having diabetes or stroke.

## Features
- **Diabetes Prediction:** Predict the likelihood of diabetes based on user inputs such as blood pressure, cholesterol levels, BMI, and other factors.
- **Stroke Prediction:** Predict the likelihood of stroke based on user inputs like gender, age, hypertension, and heart disease.

## Project Structure
- **app.py:** The main file containing the Streamlit application code.
- **saved_models:** Directory containing pre-trained machine learning models for disease prediction.
- **README.md:** Project documentation.
- **requirements.txt:** List of dependencies required to run the application.

## How to Run the Application

```bash
git clone https://github.com/yasvanthika122/Self_explainable_interface_for_disease_prediction.git
cd Self_explainable_interface_for_disease_prediction
pip install -r requirements.txt
python -m streamlit run app.py
