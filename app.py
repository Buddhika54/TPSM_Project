from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import json

app = Flask(__name__)

model = pickle.load(open('ml_model.pkl', 'rb'))

with open('results.json', 'r') as f:
    results = json.load(f)

features = [
    'Projects_Handled',
    'Training_Hours',
    'Team_Size',
    'Work_Hours_Per_Week',
    'Overtime_Hours',
    'Employee_Satisfaction_Score',
    'Sick_Days',
    'Years_At_Company'
]

@app.route('/')
def home():
    return render_template('index.html', results=results)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    employee = pd.DataFrame([{
        'Projects_Handled'            : float(data['projects']),
        'Training_Hours'              : float(data['training']),
        'Team_Size'                   : float(data['teamsize']),
        'Work_Hours_Per_Week'         : float(data['workhours']),
        'Overtime_Hours'              : float(data['overtime']),
        'Employee_Satisfaction_Score' : float(data['satisfaction']),
        'Sick_Days'                   : float(data['sickdays']),
        'Years_At_Company'            : float(data['years'])
    }])

    prediction = model.predict(employee)[0]
    probability = model.predict_proba(employee)[0]

    return jsonify({
        'result' : 'TRUE' if prediction == 1 else 'FALSE',
        'prob_true' : round(float(probability[1])*100,1),
        'prob_false' : round(float(probability[0])*100, 1)
    })

if __name__ == '__main__':
    app.run(debug=True)