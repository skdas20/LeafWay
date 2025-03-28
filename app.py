import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template, url_for, send_from_directory
import os

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Disable caching for static files
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# --- Configuration ---
MODEL_PATH = 'models/eco_predictor_model.joblib'
DATASET_PATH = 'models/vehicle_data.csv'
EXTERNAL_LANDING_PAGE = 'templates/index_home.html'
# --------------------

# Load the trained model pipeline
print(f"Loading model from {MODEL_PATH}...")
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please run the train_model.py script first to generate the model file.")
    model = None # Set model to None if loading fails
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load dataset to get unique values for dropdowns
try:
    df = pd.read_csv(DATASET_PATH)
    ROUTE_TYPES = sorted(df['Route Type'].unique().tolist())
    TIMES_OF_DAY = sorted(df['Time of Day'].unique().tolist())
    print(f"Found Route Types: {ROUTE_TYPES}")
    print(f"Found Times of Day: {TIMES_OF_DAY}")
except Exception as e:
    print(f"Warning: Could not load dataset {DATASET_PATH} to get dropdown values. Using defaults. Error: {e}")
    # Provide default values if dataset loading fails
    ROUTE_TYPES = ['Urban', 'Rural']
    TIMES_OF_DAY = ['Morning', 'Afternoon', 'Evening', 'Night'] # Added Night just in case

# CO2 emissions mapping (from the original script)
vehicle_to_co2 = {
    'Bicycle': 0,
    'Bike': 5,
    'Bus': 80,
    'Car': 180,
    'Walking': 0
}

def predict_transportation(distance, route_type, time_of_day):
    """
    Predicts transportation modes using the loaded model.
    """
    if model is None:
        return {"error": "Model not loaded. Please check server logs."}

    try:
        # Create a DataFrame with the input data matching the training format
        input_data = pd.DataFrame({
            'Distance (km)': [float(distance)],
            'Route Type': [route_type],
            'Time of Day': [time_of_day]
        })

        # Make predictions using the pipeline (handles preprocessing)
        predicted_mode = model.predict(input_data)[0]

        # Get probability estimates
        prediction_proba = model.predict_proba(input_data)[0]

        # Get all possible vehicle types from the model's classes_ attribute
        vehicle_types = model.classes_

        # Create a dictionary of modes and their probabilities
        mode_probabilities = dict(zip(vehicle_types, prediction_proba))

        # Sort modes by probability (highest first)
        sorted_modes = sorted(mode_probabilities.items(), key=lambda item: item[1], reverse=True)

        # Calculate CO2 emissions for all modes based on the input distance
        emissions_by_mode = {}
        for mode in vehicle_types:
            co2_per_km = vehicle_to_co2.get(mode, -1) # Use .get for safety, default to -1 if mode not in dict
            if co2_per_km != -1:
                emissions_by_mode[mode] = round(co2_per_km * float(distance), 2)
            else:
                emissions_by_mode[mode] = 'N/A' # Indicate if CO2 data is missing

        # Prepare the result dictionary
        result = {
            'predicted_mode': predicted_mode,
            'confidence': round(mode_probabilities.get(predicted_mode, 0), 2),
            'all_modes_ranked': [(mode, round(prob, 2)) for mode, prob in sorted_modes],
            'emissions_by_mode': emissions_by_mode,
            'input_distance': distance,
            'input_route': route_type,
            'input_time': time_of_day
        }
        return result

    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": f"Prediction failed. Error: {e}"}


@app.route('/', methods=['GET'])
def landing_page():
    """Serves the landing page."""
    try:
        return render_template('index_home.html')
    except Exception as e:
        print(f"Warning: Could not render landing page. Error: {e}")
        return render_template('start.html')


@app.route('/start', methods=['GET'])
def start_page():
    """Renders the start page as a fallback."""
    return render_template('start.html')


@app.route('/input', methods=['GET'])
def index():
    """Renders the input form page."""
    if model is None:
         # Optionally render an error page or message if model isn't loaded
         return "Error: Model could not be loaded. Please check the server logs.", 500
    return render_template('index.html', route_types=ROUTE_TYPES, times_of_day=TIMES_OF_DAY)


@app.route('/predict', methods=['POST'])
def predict():
    """Handles the form submission and displays prediction results."""
    if model is None:
         return "Error: Model could not be loaded. Please check the server logs.", 500

    try:
        # Get data from form
        distance = request.form['distance']
        route_type = request.form['route_type']
        time_of_day = request.form['time_of_day']

        # Basic validation
        if not distance or not route_type or not time_of_day:
            return "Error: All fields are required.", 400
        try:
            dist_float = float(distance)
            if dist_float <= 0:
                 raise ValueError("Distance must be positive.")
        except ValueError:
            return "Error: Distance must be a positive number.", 400

        # Call the prediction function
        prediction_result = predict_transportation(dist_float, route_type, time_of_day)

        if "error" in prediction_result:
             return f"Error during prediction: {prediction_result['error']}", 500

        # Render the results page
        return render_template('results.html', prediction=prediction_result)

    except Exception as e:
        print(f"Error in /predict route: {e}")
        return f"An unexpected error occurred: {e}", 500


# For local development
if __name__ == '__main__':
    # Check if model is loaded before running
    if model is None:
        print("CRITICAL ERROR: Model is not loaded. Flask app cannot start.")
    else:
        print("Starting Flask server...")
        # Use host='0.0.0.0' to make it accessible on the network if needed
        # debug=True is useful for development but should be False in production
        app.run(debug=True)

# This is for Vercel deployment - Vercel uses the app object directly
# No need to call app.run() as Vercel handles that
