from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Telco Churn Prediction API"
    }


def test_predict_endpoint(monkeypatch):

    def mock_prediction(customer_data):
        return [1], "Likely to churn"

    monkeypatch.setattr(
        "src.api.main.make_prediction",
        mock_prediction
    )

    customer_data = {
        "Gender": "Male",
        "SeniorCitizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "Tenure": 12,
        "MonthlyCharges": 50.0,
        "TotalCharges": 600.0,
    }

    response = client.post(
        "/predict",
        json=customer_data
    )

    assert response.status_code == 200
    assert response.json() == [1]