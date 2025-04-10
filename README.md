# Heart Disease Prediction System

This project implements a machine learning system to predict heart disease risk based on medical attributes. It includes data preprocessing, model training, and a user-friendly web interface built with Streamlit.

## Project Structure

```
heart-disease-prediction/
├── heart_disease_data.csv      # Dataset file
├── preprocess.py               # Data preprocessing script
├── train_model.py              # Model training script
├── app.py                      # Streamlit web application
├── requirements.txt            # Package dependencies
├── heart_disease_model.pkl     # Saved trained model (generated)
├── preprocessor.pkl            # Saved data preprocessor (generated)
├── feature_info.pkl            # Saved feature information (generated)
└── README.md                   # Project documentation
```

## Features

- Data preprocessing and exploratory data analysis
- Machine learning model (Random Forest Classifier) for heart disease prediction
- Interactive web interface for inputting patient data and receiving predictions
- Model evaluation metrics and feature importance analysis

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd heart-disease-prediction
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Preprocess the Data
```
python preprocess.py
```
This will load the dataset, preprocess it, and save the preprocessor for later use.

### Step 2: Train the Model
```
python train_model.py
```
This will train a Random Forest Classifier on the preprocessed data, evaluate its performance, and save the trained model.

### Step 3: Run the Streamlit Application
```
streamlit run app.py
```
This will start the Streamlit server and open the web application in your default browser.

## Using the Application

1. Enter patient attributes in the web form.
2. Click the "Predict" button to see the heart disease risk prediction.
3. The result will show whether the patient is at high risk of heart disease or not, along with a confidence percentage.

## Model Information

- **Algorithm**: Random Forest Classifier
- **Features**: Age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG results, maximum heart rate, exercise-induced angina, ST depression, slope of the peak exercise ST segment, number of major vessels colored by fluoroscopy, and thalassemia.
- **Target**: Binary classification (0: No heart disease, 1: Heart disease)

## Dataset

The dataset used in this project contains 303 patient records with various medical attributes. Each record includes 13 features and a target variable indicating the presence or absence of heart disease.

## Disclaimer

This application is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.