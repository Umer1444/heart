import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocess import load_and_process_data

def train_and_evaluate_model():
    """
    Train RandomForestClassifier on the preprocessed data and evaluate its performance.
    Also saves the trained model and preprocessor for later use in the app.
    """
    # Load and preprocess the data
    X_train, X_test, y_train, y_test, preprocessor = load_and_process_data()
    
    # Apply the preprocessing to the training and test data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Initialize and train the model
    print("\nTraining RandomForest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_processed, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_processed)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Save the trained model and preprocessor
    joblib.dump(model, 'heart_disease_model.pkl')
    joblib.dump(preprocessor, 'preprocessor.pkl')
    print("\n✅ Model saved as 'heart_disease_model.pkl'")
    print("✅ Preprocessor saved as 'preprocessor.pkl'")
    
    # Feature importance (optional)
    if hasattr(model, 'feature_importances_'):
        # Get feature names after transformation
        feature_info = joblib.load('feature_info.pkl')
        numerical_features = feature_info['numerical_features']
        categorical_features = feature_info['categorical_features']
        
        # Get transformed feature names
        transformed_features = preprocessor.transformers_[0][1].get_feature_names_out(numerical_features).tolist()
        transformed_features += preprocessor.transformers_[1][1].get_feature_names_out(categorical_features).tolist()
        
        # Create a DataFrame of feature importances
        feature_importance = pd.DataFrame({
            'Feature': transformed_features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))
    
    return model, preprocessor

if __name__ == "__main__":
    model, preprocessor = train_and_evaluate_model()
