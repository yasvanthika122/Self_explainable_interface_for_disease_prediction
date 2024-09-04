import joblib
import pickle
import os
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

# Set the working directory
working_dir = os.path.dirname(os.path.realpath(__file__))

# loading the saved diabetes model
try:
    diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
except Exception as e:
    st.error(f"Error loading diabetes model: {e}")

# Try loading the stroke model
try:
    stroke_model = pickle.load(open(f'{working_dir}/saved_models/stroke_model.sav', 'rb'))
except Exception as pickle_error:
    try:
        stroke_model = joblib.load(f'{working_dir}/saved_models/stroke_model.sav')
    except Exception as joblib_error:
        st.error(f"Error loading stroke model: {pickle_error} or {joblib_error}")

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Stroke Prediction'],
                           icons=['activity', 'thermometer'],
                           default_index=0)

# Mapping new labels to original model features
label_mapping = {
    'HighBP': 'Pregnancies',
    'HighChol': 'Glucose',
    'CholCheck': 'BloodPressure',
    'BMI': 'SkinThickness',
    'Smoker': 'Insulin',
    'HeartDiseaseorAttack': 'BMI',
    'PhysActivity': 'DiabetesPedigreeFunction',
    'Fruits': 'Age'
}

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        HighBP = st.text_input('High Blood Pressure')

    with col2:
        HighChol = st.text_input('High Cholesterol')

    with col3:
        CholCheck = st.text_input('Cholesterol Check')

    with col1:
        BMI = st.text_input('Body Mass Index (BMI)')

    with col2:
        Smoker = st.text_input('Smoker (0 or 1)')

    with col3:
        HeartDiseaseorAttack = st.text_input('Heart Disease or Heart Attack (0 or 1)')

    with col1:
        PhysActivity = st.text_input('Physical Activity (0 or 1)')

    with col2:
        Fruits = st.text_input('Stroke (0 or 1)')

    # prediction code
    diabetes_diagnosis = ''

    # create button for prediction
    if st.button('Diabetes Test Result'):
        # Prepare input based on original model features
        input_data = [
            HighBP,   # Pregnancies
            HighChol, # Glucose
            CholCheck, # Blood Pressure
            BMI,      # Skin Thickness
            Smoker,   # Insulin
            HeartDiseaseorAttack, # BMI
            PhysActivity, # Diabetes Pedigree Function
            Fruits    # Age
        ]

        # Convert input to appropriate dtype for prediction
        input_data = np.array(input_data).astype(float).reshape(1, -1)

        diabetes_prediction = diabetes_model.predict(input_data)

        if diabetes_prediction[0] == 1:
            diabetes_diagnosis = 'The person is likely to have diabetes'
        else:
            diabetes_diagnosis = 'The person is not likely to have diabetes'

    st.success(diabetes_diagnosis)

# Stroke Prediction Page
if selected == 'Stroke Prediction':
    st.title('Stroke Prediction using ML')

    # Collect all 10 inputs from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.text_input('Gender (1 for Male, 0 for Female)')

    with col2:
        age = st.text_input('Age')

    with col3:
        hypertension = st.text_input('Hypertension (1 for Yes, 0 for No)')

    with col1:
        heart_disease = st.text_input('Heart Disease (1 for Yes, 0 for No)')

    with col2:
        ever_married = st.text_input('Ever Married (1 for Yes, 0 for No)')

    with col3:
        work_type = st.text_input('Work Type')

    with col1:
        Residence_type = st.text_input('Residence Type (1 for Urban, 0 for Rural)')

    with col2:
        avg_glucose_level = st.text_input('Average Glucose Level')

    with col3:
        bmi = st.text_input('BMI value')

    with col1:
        smoking_status = st.text_input('Smoking Status (1 for Never smoked, 2 for Formerly smoked, 3 for Smokes)')

    # Create a button for prediction
    if st.button('Stroke Test Result'):
        if stroke_model:
            # Extract only the 4 features that the model was trained on
            input_data = np.array([[gender, age, hypertension, heart_disease]])
            
            # Make the prediction using only those 4 features
            stroke_prediction = stroke_model.predict(input_data)
            
            # Display the prediction result
            diagnosis = 'The person is likely to have a stroke' if stroke_prediction[0] == 1 else 'The person is not likely to have a stroke'
            st.success(diagnosis)
        else:
            st.error("Stroke model not loaded")
