import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import os

# 1. Generate Synthetic Data
def generate_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'CustomerID': range(1, n_samples + 1),
        'Age': np.random.randint(18, 70, n_samples),
        'Tenure': np.random.randint(0, 72, n_samples),
        'MonthlyCharges': np.random.uniform(20, 120, n_samples),
        'TotalCharges': np.random.uniform(100, 8000, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'Churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    # Add some correlation for realistic modeling
    df = pd.DataFrame(data)
    df.loc[df['Contract'] == 'Month-to-month', 'Churn'] = np.random.choice([0, 1], len(df[df['Contract'] == 'Month-to-month']), p=[0.5, 0.5])
    df.loc[df['MonthlyCharges'] > 100, 'Churn'] = np.random.choice([0, 1], len(df[df['MonthlyCharges'] > 100]), p=[0.4, 0.6])
    return df

# 2. Preprocess Data
def preprocess_data(df):
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['Gender', 'Contract', 'PaymentMethod']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    # Define features and target
    X = df.drop(['CustomerID', 'Churn'], axis=1)
    y = df['Churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X.columns

# 3. Train Model
def train_model(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model

# 4. Main Execution
if __name__ == "__main__":
    print("Generating synthetic dataset...")
    df = generate_data()
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    
    print("Training XGBoost model...")
    model = train_model(X_train, y_train)
    
    print("\nModel Evaluation:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Save results for the portfolio page (mock results)
    with open('model_stats.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred))
    
    print("\nResults saved to model_stats.txt")
