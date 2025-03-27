# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib # Added for saving the model
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
DATASET_PATH = 'vehicle_data.csv' # Make sure this file is in the same directory
MODEL_SAVE_PATH = 'eco_predictor_model.joblib'
# --------------------

print(f"Loading dataset from: {DATASET_PATH}")
try:
    df = pd.read_csv(DATASET_PATH)
    print("Dataset loaded successfully.")
    # Display the first few rows of the dataset
    print("Dataset Preview:")
    print(df.head())
except FileNotFoundError:
    print(f"Error: Dataset file not found at {DATASET_PATH}")
    print(f"Please ensure '{DATASET_PATH}' is in the same directory as this script.")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()


# -------------------------------------------------------------------------
# Data Preprocessing
# -------------------------------------------------------------------------
print("\nStarting Data Preprocessing...")
# Define features (X) and target variable (y)
X = df[['Distance (km)', 'Route Type', 'Time of Day']]
y = df['Vehicle Type']

# Get unique values for categorical features (useful for Flask app later)
route_types = X['Route Type'].unique().tolist()
times_of_day = X['Time of Day'].unique().tolist()
print(f"Unique Route Types found: {route_types}")
print(f"Unique Times of Day found: {times_of_day}")


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Define preprocessing for numerical and categorical features
numerical_features = ['Distance (km)']
categorical_features = ['Route Type', 'Time of Day']

# Create a preprocessing pipeline
# Use handle_unknown='ignore' in OneHotEncoder in case new categories appear in prediction
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features) # sparse=False might be needed for some models/versions
    ],
    remainder='passthrough' # Keep other columns if any (though we only selected 3)
)
print("Preprocessor created.")

# -------------------------------------------------------------------------
# Model Training and Selection
# -------------------------------------------------------------------------
print("\nStarting Model Training and Selection...")
# Define models to evaluate
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42) # probability=True needed for predict_proba
}

# Train and evaluate each model
results = {}
best_accuracy = -1
best_model_name = None
best_pipeline = None

for name, model in models.items():
    print(f"Training {name}...")

    # Create a pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {'model': pipeline, 'accuracy': accuracy}
    print(f"{name} - Accuracy: {accuracy:.4f}")

    # Keep track of the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name
        best_pipeline = pipeline

print(f"\nBest Initial Model: {best_model_name} with accuracy {best_accuracy:.4f}")

# -------------------------------------------------------------------------
# Model Hyperparameter Tuning for the Best Model
# -------------------------------------------------------------------------
print(f"\nStarting Hyperparameter Tuning for {best_model_name}...")

# Define hyperparameter grids for each model type
param_grid = {}
if best_model_name == 'Random Forest':
    param_grid = {
        'classifier__n_estimators': [50, 100, 200], # Reduced options for faster tuning
        'classifier__max_depth': [None, 10],
        'classifier__min_samples_split': [2, 5]
    }
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__max_depth': [3, 5]
    }
elif best_model_name == 'SVM':
    param_grid = {
        'classifier__C': [0.1, 1, 10],
        'classifier__gamma': ['scale', 'auto'],
        'classifier__kernel': ['rbf'] # Limiting kernel for faster tuning
    }

# Perform Grid Search CV for the best model pipeline
if param_grid: # Only tune if a grid is defined
    grid_search = GridSearchCV(best_pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1) # cv=3 for speed
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters
    print(f"\nBest parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    # Evaluate the tuned model
    tuned_model = grid_search.best_estimator_
    y_pred_tuned = tuned_model.predict(X_test)
    tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
    print(f"Tuned Model Test Accuracy: {tuned_accuracy:.4f}")
else:
    print("No hyperparameter grid defined for the best model, using the initial best model.")
    tuned_model = best_pipeline # Use the initial best if no tuning happened
    tuned_accuracy = best_accuracy

# -------------------------------------------------------------------------
# Save the Final Model
# -------------------------------------------------------------------------
print(f"\nSaving the final tuned model to {MODEL_SAVE_PATH}...")
try:
    joblib.dump(tuned_model, MODEL_SAVE_PATH)
    print(f"Model saved successfully with accuracy: {tuned_accuracy:.4f}")
except Exception as e:
    print(f"Error saving model: {e}")

print("\nTraining script finished.")
