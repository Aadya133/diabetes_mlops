from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_prediction():
    payload = {
        "gender": "Female",
        "age": 45,
        "hypertension": 0,
        "heart_disease": 0,
        "smoking_history": "never",
        "bmi": 26.5,
        "HbA1c_level": 5.7,
        "blood_glucose_level": 120
    }

    response = client.post("/Prediction", json=payload)

    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in [0, 1]
    assert 0 <= result["class_1_probability"] <= 1