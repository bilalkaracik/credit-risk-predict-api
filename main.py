import joblib
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = Flask(__name__)


data = pd.DataFrame({
    'person_home_ownership': ['OWN', 'MORTGAGE', 'RENT', 'OTHER', 'OWN' , 'OTHER', 'OWN'],
    'loan_intent': ['EDUCATION', 'DEBTCONSOLIDATION', 'VENTURE', 'MEDICAL' , 'PERSONAL', 'HOMEIMPROVEMENT' , 'HOMEIMPROVEMENT' ],
    'loan_grade': ['A' , 'B' , 'C' , 'D' , 'E' , 'F' , 'G'],
    'cb_person_default_on_file': ['Y', 'N', 'N', 'N', 'Y' ,'N' , 'Y']
})

# Kategorik değişkenlerin listesi
categorical_variables = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
label_encoders = {}

for col in categorical_variables:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])


model = joblib.load("predict_model_v2.joblib")

@app.route('/')
def root():
    return "Credit Risk Predict"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_json = request.json  

        
        input_data = pd.DataFrame(input_json)
        for col in categorical_variables:
            input_data[col] = label_encoders[col].transform(input_data[col])

        input_data["Ratio of Age to Income"] = input_data["person_age"] / input_data["person_income"]
        input_data['Ratio of Loan Amount to Age'] = input_data["loan_amnt"] / input_data["person_age"]
        input_data['Ratio'] = (input_data["loan_amnt"] * input_data['loan_int_rate']) / input_data['person_income']
        input_data["Ratio of Grade to Amount"] = input_data["loan_grade"] / input_data["loan_amnt"] 

        prediction = model.predict(input_data)  

        response = {
            'Loan Status': prediction.tolist()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False)
