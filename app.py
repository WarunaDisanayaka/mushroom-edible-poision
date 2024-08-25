from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import xgboost as xgb

app = Flask(__name__)

# Enable CORS for the entire app
CORS(app)

# Load the saved model
model = xgb.Booster()
model.load_model('mushroom_xgb_model.json')

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        
        print(data)

        # Create DataFrame from the received JSON data
        new_data = pd.DataFrame(data)
        
        # Handle missing values
        new_data['stem-width'] = new_data['stem-width'].fillna('missing')
        
        # Define categorical columns
        category_cols = [
            'cap-shape', 'cap-surface', 'cap-color', 'does-bruise-or-bleed', 
            'gill-attachment', 'gill-spacing', 'gill-color', 'stem-root', 
            'stem-surface', 'stem-color', 'veil-type', 'veil-color', 
            'has-ring', 'ring-type', 'spore-print-color', 'habitat', 'season'
        ]
        
        # Convert categorical columns to numeric codes
        for category in category_cols:
            if category in new_data.columns:
                new_data[category] = new_data[category].astype('category').cat.codes
        
        # Ensure all columns are numeric
        new_data = new_data.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Convert DataFrame to DMatrix
        D_new = xgb.DMatrix(new_data)
        
        # Get predictions
        preds = model.predict(D_new)
        predictions = [round(value) for value in preds]
        
        # Return the predictions as JSON
        return jsonify(predictions=predictions)
    
    except Exception as e:
        return jsonify(error=str(e)), 400

if __name__ == '__main__':
    app.run(debug=True, port=8000)  # Change the port here
