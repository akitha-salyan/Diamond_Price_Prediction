import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create Flask app
app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route (HTML form)
@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predict diamond price based on user input
    '''
    # Get values from form and convert to float
    features = [float(x) for x in request.form.values()]
    final_features = np.array([features])

    # Make prediction
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    return render_template(
        'index.html',
        prediction_text=f'Price of Diamond should be $ {output}'
    )

# API route
@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    Predict price using API (JSON input)
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    return jsonify(prediction[0])

# Run app
if __name__ == "__main__":
    app.run(debug=True)