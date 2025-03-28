# LeafWay - Eco-friendly Transportation Predictor

![LeafWay Logo](/static/leaf.png)

## Overview

LeafWay is an eco-friendly transportation predictor web application designed to help users make smarter, greener transportation choices. By analyzing factors such as distance, route type, and time of day, LeafWay recommends the most environmentally friendly transportation options for your journey and provides detailed CO2 emission estimates.

## Features

- **Smart Transportation Predictions**: Utilizes machine learning to predict the most suitable transportation mode for your trip
- **CO2 Emission Tracking**: Provides detailed CO2 emission estimates for different transportation methods
- **Cost-Effective Suggestions**: Helps identify budget-friendly transportation alternatives
- **User-Friendly Interface**: Intuitive design with responsive layout for both desktop and mobile devices
- **Dark/Light Theme Toggle**: Personalized viewing experience with theme preference memory
- **Interactive Results**: Visualizes transportation options ranked by suitability and environmental impact

## Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript
- **Machine Learning**: scikit-learn, pandas, numpy, joblib
- **Data Processing**: pandas for data manipulation and analysis

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/leafway.git
cd leafway
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Ensure the model file is available:
```
python train_model.py
```

## Usage

1. Start the Flask server:
```
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

3. Use the "Get Started" button to access the prediction form.

4. Enter your trip details:
   - Distance (in kilometers)
   - Route type (Urban/Rural)
   - Time of day

5. Click "Predict Transportation" to receive personalized recommendations.

## Project Structure

```
LeafWay/
├── app.py                 # Main Flask application file
├── train_model.py         # Script to train the prediction model
├── eco_predictor_model.joblib  # Trained machine learning model
├── vehicle_data.csv       # Dataset for training the model
├── index_home.html        # Landing page template
├── requirements.txt       # Python dependencies
├── static/                # Static files (images, etc.)
│   ├── leaf.png           # Favicon/logo
│   └── Untitled_design.jpg # Main image
└── templates/             # Flask HTML templates
    ├── index.html         # Input form page
    ├── results.html       # Prediction results page
    └── start.html         # Start page
```

## Model

The transportation mode prediction model is trained on a dataset that includes:
- Distance of the journey (in kilometers)
- Route type (Urban or Rural)
- Time of day (Morning, Afternoon, Evening)

The model predicts the most suitable transportation mode from the following options:
- Walking
- Bicycle
- Bike (Motorbike)
- Bus
- Car

Each mode is associated with CO2 emissions data to help users make environmentally conscious decisions.

## Customization

- **Adding New Transportation Modes**: Update the `vehicle_to_co2` dictionary in `app.py` with new modes and their emissions data
- **Changing the Theme**: Modify the CSS variables in the `:root` and `.dark-theme` selectors in the HTML files
- **Updating Predictions**: Enhance the dataset and retrain the model using `train_model.py`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Created by Rishav kr.
- Powered by Creativity

---

© 2023 LeafWay. All rights reserved.