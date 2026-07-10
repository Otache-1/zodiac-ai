import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import ast
import joblib

def train_ai_model():
    print("Loading processed molecular feature data...")
    # Read our processed data file
    df = pd.read_csv("malaria_processed.csv")
    
    # Convert the string representation of lists back into actual Python lists
    X = np.array([ast.literal_eval(fp) for fp in df['fingerprint']])
    y = df['activity'].values
    
    print(f"Feature matrix shape: {X.shape} (Molecules, Structural Bits)")
    print(f"Target array shape: {y.shape} (Activity labels)")
    
    # Split the data into a training set and a testing set (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nInitializing Random Forest Classifier...")
    # Create the Machine Learning model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    print("Training the AI engine on molecular structures...")
    model.fit(X_train, y_train)
    
    # Evaluate how well our model performs on the test set
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\nModel Training Complete!")
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    
    # Save the trained AI model to a physical file so we can use it to screen new molecules later
    model_filename = "zodiac_malaria_model.pkl"
    joblib.dump(model, model_filename)
    print(f"Saved trained brain model as '{model_filename}'")

if __name__ == "__main__":
    train_ai_model()