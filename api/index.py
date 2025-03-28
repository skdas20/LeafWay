from flask import Flask, request, jsonify, send_from_directory
import os
import sys

# Add the current directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize Flask app
app = Flask(__name__)

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
    return jsonify({
        "status": "ok",
        "message": "Welcome to LeafWay API",
        "documentation": "Visit /api/health for API information"
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handles prediction requests."""
    try:
        # Get data from form or JSON
        data = request.get_json() if request.is_json else request.form
        
        distance = data.get('distance')
        route_type = data.get('route_type')
        time_of_day = data.get('time_of_day')

        # Basic validation
        if not distance or not route_type or not time_of_day:
            return jsonify({"error": "All fields are required"}), 400
        
        try:
            dist_float = float(distance)
            if dist_float <= 0:
                raise ValueError("Distance must be positive.")
        except ValueError:
            return jsonify({"error": "Distance must be a positive number"}), 400

        # Get prediction
        prediction_result = predict_eco_transport(dist_float, route_type, time_of_day)
        
        # Return JSON response
        return jsonify(prediction_result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "ok", 
        "message": "LeafWay API is running",
        "version": "1.0.0",
        "python_version": sys.version,
        "routes": {
            "api_root": "/",
            "health_check": "/api/health",
            "prediction": "/api/predict (POST)",
            "info": "/api/info"
        },
        "sample_request": {
            "url": "/api/predict",
            "method": "POST",
            "body": {
                "distance": 5,
                "route_type": "Urban",
                "time_of_day": "Morning"
            }
        }
    }), 200

@app.route('/api/info', methods=['GET'])
def info():
    """Returns information about the API"""
    return jsonify({
        "name": "LeafWay API",
        "description": "Eco-friendly transportation recommendation API",
        "route_types": ROUTE_TYPES,
        "times_of_day": TIMES_OF_DAY,
        "transport_modes": list(vehicle_co2.keys())
    })

# Export the Flask app for Vercel
app.debug = False 