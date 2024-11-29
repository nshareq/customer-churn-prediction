import requests
import json

def test_prediction_api():
    # API endpoint
    url = "http://localhost:8080/api/predict"
    
    # Test data based on the dataset structure
    test_data = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.70,
        "TotalCharges": 1695.40
    }
    
    # Make POST request
    try:
        response = requests.post(url, json=test_data)
        print("Status Code:", response.status_code)
        print("Response:", response.json())
        
        # Basic validation
        result = response.json()
        assert "prediction" in result, "Prediction missing in response"
        assert "churn_probability" in result, "Probability missing in response"
        print("\nTest passed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    test_prediction_api()