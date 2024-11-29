from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from src.pipelines.prediction_pipeline import PredictionPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('form.html')
    
    else:
        try:
            data = CustomData(
                gender=request.form.get('gender'),
                SeniorCitizen=int(request.form.get('SeniorCitizen')),
                Partner=request.form.get('Partner'),
                Dependents=request.form.get('Dependents'),
                tenure=int(request.form.get('tenure')),
                PhoneService=request.form.get('PhoneService'),
                MultipleLines=request.form.get('MultipleLines'),
                InternetService=request.form.get('InternetService'),
                OnlineSecurity=request.form.get('OnlineSecurity'),
                OnlineBackup=request.form.get('OnlineBackup'),
                DeviceProtection=request.form.get('DeviceProtection'),
                TechSupport=request.form.get('TechSupport'),
                StreamingTV=request.form.get('StreamingTV'),
                StreamingMovies=request.form.get('StreamingMovies'),
                Contract=request.form.get('Contract'),
                PaperlessBilling=request.form.get('PaperlessBilling'),
                PaymentMethod=request.form.get('PaymentMethod'),
                MonthlyCharges=float(request.form.get('MonthlyCharges')),
                TotalCharges=float(request.form.get('TotalCharges'))
            )
            
            df = data.get_data_as_dataframe()
            prediction_pipeline = PredictionPipeline()
            pred, prob = prediction_pipeline.predict(df)
            
            return render_template('result.html', 
                                 prediction=pred[0], 
                                 probability=round(prob[0][1] * 100, 2))

        except Exception as e:
            return jsonify({'error': str(e)})

@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        data = request.json
        df = pd.DataFrame(data, index=[0])
        prediction_pipeline = PredictionPipeline()
        pred, prob = prediction_pipeline.predict(df)
        
        return jsonify({
            'prediction': int(pred[0]),
            'churn_probability': float(prob[0][1]),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'failed'
        })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)