from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

app = Flask(__name__)

model = pickle.load(open('xgboost.pkl', 'rb'))

gender_mapping = {"female": 0, "male": 1, "f": 0, "m": 1}

# Define mapping for treatment_duration
treatment_duration_mapping = {
    "2 -< 5yrs": 1,
    "1 -< 2yrs": 0,
    "6 months -< 1yr": 3,
    "< 6 months": 4,
    "5yrs and above": 2
}

# Define mapping for arv_adherence
arv_adherence_mapping = {
    "good": 1,
    "fair": 0,
    "poor": 2
}

# Define mapping for Indication_for_VL_Testing
indication_mapping = {
    "routine monitoring": 5,
    "repeat viral load": 4,
    "repeat (after iac)": 3,
    "12 months after art initiation": 0,
    "6 months after art initiation": 2,
    "1st anc for pmtct": 1,
    "suspected treatment failure": 6
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Checking for missing fields
        required_fields = ['gender', 'encounter_date', 'art_start_date', 'date_birth', 'regimen_line', 'Indication_for_VL_Testing', 'arv_adherence']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

        # Calculate age from encounter_date and art_start_date
        try:
            encounter_date = datetime.strptime(data['encounter_date'], '%Y-%m-%d')
            art_start_date = datetime.strptime(data['art_start_date'], '%Y-%m-%d')
            date_birth = datetime.strptime(data['date_birth'], '%Y-%m-%d')
            age = (encounter_date - date_birth).days // 365  # Calculate age in years
            treatment_time = (encounter_date - art_start_date).days
        except ValueError:
            return jsonify({"error": "Invalid date format. Use 'YYYY-MM-DD'."}), 400

        # Calculate treatment_duration based on treatment_time
        if treatment_time < 183:
            treatment_duration = "6 months -< 1yr"
        elif 183 <= treatment_time < 365:
            treatment_duration = "1 -< 2yrs"
        elif 365 <= treatment_time < 730:
            treatment_duration = "2 -< 5yrs"
        elif 730 <= treatment_time:
            treatment_duration = "5yrs and above"
        else:
            treatment_duration = "< 6 months"

        # Pre-process data
        input_data = {
            'gender': gender_mapping.get(data['gender'].lower(), -1),  # default to -1 if not found
            ' age': age,  # Use calculated age
            'treatment_duration': treatment_duration_mapping.get(treatment_duration, -1),  # Use mapped value
            'regimen_line': data['regimen_line'],
            'Indication_for_VL_Testing': indication_mapping.get(data['Indication_for_VL_Testing'].lower(), -1),  # Use mapped value
            'arv_adherence': arv_adherence_mapping.get(data['arv_adherence'].strip().lower(), -1)  # Use mapped value
        }

        # Checking if gender mapping was successful
        if input_data['gender'] == -1:
            return jsonify({"error": "Invalid gender provided. Use 'male' or 'female'."}), 400

        # Check if treatment_duration mapping was successful
        if input_data['treatment_duration'] == -1:
            return jsonify({"error": "Invalid treatment_duration provided."}), 400

        # Check if Indication_for_VL_Testing mapping was successful
        if input_data['Indication_for_VL_Testing'] == -1:
            return jsonify({"error": "Invalid Indication_for_VL_Testing provided.It should either be Routine Monitoring,Repeat Viral Load,Repeat (after IAC),2 months after ART initiation,6 months after ART initiation,6 months after ART initiation,1st ANC For PMTCT,Suspected Treatment Failure."}), 400

        # Check if arv_adherence mapping was successful
        if input_data['arv_adherence'] == -1:
            return jsonify({"error": "Invalid arv_adherence provided. It should either be Good,Fair,or Poor."}), 400

        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)

        # Post-process the results
        if prediction == 1:
            result = "Client: UnSuppressed"
        else:
            result = "Client: Suppressed"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
