import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def load_and_process_data(file_path='heart_disease_data.csv'):
    """
    Load and preprocess the heart disease dataset
    """
    # Load the dataset
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Display basic information
    print("\nDataset Information:")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nChecking for missing values:")
    print(df.isnull().sum())
    
    print("\nColumn names:", df.columns.tolist())
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Identify categorical and numerical features
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    # Create preprocessing pipelines for both categorical and numerical data
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Save the feature names and their respective types for later use
    feature_info = {
        'categorical_features': categorical_features,
        'numerical_features': numerical_features
    }
    
    joblib.dump(feature_info, 'feature_info.pkl')
    joblib.dump(preprocessor, 'preprocessor.pkl')
    
    return X_train, X_test, y_train, y_test, preprocessor

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor = load_and_process_data()
    print("\nPreprocessing completed. Preprocessor saved as 'preprocessor.pkl'")
    print("Feature information saved as 'feature_info.pkl'")