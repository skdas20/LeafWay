from flask import Flask, request, render_template, url_for, jsonify
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

# Constants for the app - using fixed values instead of loading from model
ROUTE_TYPES = ['Urban', 'Rural']
TIMES_OF_DAY = ['Morning', 'Afternoon', 'Evening', 'Night']

# Mock CO2 emissions data - simplified from the original app
vehicle_co2 = {
    'Bicycle': 0,
    'Bike': 5,
    'Bus': 80,
    'Car': 180,
    'Walking': 0
}

def predict_eco_transport(distance, route_type, time_of_day):
    """
    Simple prediction logic without ML model
    """
    # Simple distance-based rules to determine best transport mode
    if float(distance) < 2:
        primary_mode = "Walking"
        confidence = 0.92
    elif float(distance) < 5:
        primary_mode = "Bicycle"
        confidence = 0.85
    elif float(distance) < 15:
        if route_type == "Urban":
            primary_mode = "Bus"
            confidence = 0.78
        else:
            primary_mode = "Bike"
            confidence = 0.75
    else:
        if route_type == "Urban" and time_of_day in ["Morning", "Evening"]:
            primary_mode = "Bus"
            confidence = 0.65
        else:
            primary_mode = "Car"
            confidence = 0.80
    
    # Create ranking of all modes
    all_modes = [
        ("Walking", 0.95 if float(distance) < 2 else 0.3 if float(distance) < 5 else 0.1),
        ("Bicycle", 0.85 if float(distance) < 5 else 0.4 if float(distance) < 10 else 0.15),
        ("Bike", 0.7 if 3 < float(distance) < 15 else 0.5 if float(distance) < 25 else 0.3),
        ("Bus", 0.8 if route_type == "Urban" else 0.4),
        ("Car", 0.4 if float(distance) < 10 else 0.8)
    ]
    
    # Sort by probability
    all_modes.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate CO2 emissions
    emissions = {}
    for mode in vehicle_co2.keys():
        emissions[mode] = round(vehicle_co2[mode] * float(distance), 2)
    
    # Prepare result
    result = {
        'predicted_mode': primary_mode,
        'confidence': confidence,
        'all_modes_ranked': all_modes,
        'emissions_by_mode': emissions,
        'input_distance': distance,
        'input_route': route_type,
        'input_time': time_of_day
    }
    
    return result

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
    return render_template('index.html', route_types=ROUTE_TYPES, times_of_day=TIMES_OF_DAY)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the form submission and displays prediction results."""
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

        # Get prediction
        prediction_result = predict_eco_transport(dist_float, route_type, time_of_day)
        
        # Render the results page
        return render_template('results.html', prediction=prediction_result)

    except Exception as e:
        print(f"Error in /predict route: {e}")
        return f"An unexpected error occurred: {e}", 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "ok", "message": "LeafWay API is running"}), 200

# For local development
if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)

# For Vercel deployment
app.debug = False
