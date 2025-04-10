import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.exceptions import NotFittedError

def load_resources():
    """Load the trained model, preprocessor and feature information"""
    model = joblib.load('heart_disease_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    feature_info = joblib.load('feature_info.pkl')
    return model, preprocessor, feature_info

def preprocess_input(user_input, preprocessor):
    """Preprocess the user input using the saved preprocessor"""
    input_df = pd.DataFrame([user_input])
    try:
        preprocessed_input = preprocessor.transform(input_df)
    except NotFittedError:
        st.error("⚠️ The preprocessor has not been fitted. Please fit it on training data and re-save.")
        st.stop()
    return preprocessed_input

def predict_heart_disease(preprocessed_input, model):
    """Make prediction using the trained model"""
    prediction = model.predict(preprocessed_input)
    probability = model.predict_proba(preprocessed_input)
    return prediction[0], probability[0]

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Heart Disease Prediction",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load resources
    try:
        model, preprocessor, feature_info = load_resources()
    except FileNotFoundError:
        st.error("Model files not found. Please run 'train_model.py' first to train and save the model and preprocessor.")
        st.stop()
    
    # App title and description
    st.title("❤️ Heart Disease Prediction App")
    st.markdown("""
    This application predicts whether a person is likely to have heart disease based on their medical attributes.
    Please fill in the patient information below and click 'Predict' to get the result.
    """)
    
    # Define the categorical features options
    sex_options = {0: 'Female', 1: 'Male'}
    cp_options = {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'}
    fbs_options = {0: 'No (≤ 120 mg/dl)', 1: 'Yes (> 120 mg/dl)'}
    restecg_options = {0: 'Normal', 1: 'ST-T Wave Abnormality', 2: 'Left Ventricular Hypertrophy'}
    exang_options = {0: 'No', 1: 'Yes'}
    slope_options = {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}
    ca_options = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4'}
    thal_options = {0: 'Normal', 1: 'Fixed Defect', 2: 'Reversible Defect', 3: 'Unknown'}
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    # Input fields for the left column
    with col1:
        st.subheader("Patient Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=45)
        sex = st.selectbox("Sex", options=list(sex_options.keys()), format_func=lambda x: sex_options[x])
        cp = st.selectbox("Chest Pain Type", options=list(cp_options.keys()), format_func=lambda x: cp_options[x])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=130)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=230)
        fbs = st.selectbox("Fasting Blood Sugar", options=list(fbs_options.keys()), format_func=lambda x: fbs_options[x])
    
    # Input fields for the right column
    with col2:
        st.subheader("Medical Test Results")
        restecg = st.selectbox("Resting ECG Results", options=list(restecg_options.keys()), format_func=lambda x: restecg_options[x])
        thalach = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina", options=list(exang_options.keys()), format_func=lambda x: exang_options[x])
        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", options=list(slope_options.keys()), format_func=lambda x: slope_options[x])
        ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=list(ca_options.keys()), format_func=lambda x: ca_options[x])
        thal = st.selectbox("Thalassemia", options=list(thal_options.keys()), format_func=lambda x: thal_options[x])
    
    # Create a user input dictionary
    user_input = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    # Add a predict button
    if st.button("Predict", type="primary"):
        with st.spinner("Analyzing medical data..."):
            time.sleep(1)
            
            # Preprocess the input and make prediction
            preprocessed_input = preprocess_input(user_input, preprocessor)
            prediction, probability = predict_heart_disease(preprocessed_input, model)
            
            # Display the prediction result
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error("⚠️ High Risk of Heart Disease")
                st.markdown(f"The model predicts a **{probability[1]*100:.1f}%** probability of heart disease.")
            else:
                st.success("✅ No Heart Disease Detected")
                st.markdown(f"The model predicts a **{probability[0]*100:.1f}%** probability of no heart disease.")
            
            # Disclaimer
            st.caption("Disclaimer: This tool is for educational purposes only and should not be used as a substitute for professional medical advice.")
    
    # Model details
    with st.expander("About this model"):
        st.write("""
        This heart disease prediction model is trained on a dataset of 303 patients. 
        
        **Features used in the model:**
        - Age: Age of the patient in years
        - Sex: Sex of the patient (0: Female, 1: Male)
        - Chest Pain Type (cp): Type of chest pain experienced
        - Resting Blood Pressure (trestbps): Resting blood pressure in mm Hg
        - Serum Cholesterol (chol): Serum cholesterol in mg/dl
        - Fasting Blood Sugar (fbs): Whether fasting blood sugar > 120 mg/dl
        - Resting Electrocardiographic Results (restecg): Results of resting ECG
        - Maximum Heart Rate (thalach): Maximum heart rate achieved
        - Exercise Induced Angina (exang): Whether exercise induced angina
        - ST Depression (oldpeak): ST depression induced by exercise relative to rest
        - Slope of the Peak Exercise ST Segment (slope): Slope of the peak exercise ST segment
        - Number of Major Vessels (ca): Number of major vessels colored by fluoroscopy
        - Thalassemia (thal): A blood disorder called thalassemia
        
        The model uses a Random Forest classifier trained on historical patient data.
        """)

if __name__ == "__main__":
    main()
