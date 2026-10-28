import joblib
import pandas as pd
from pathlib import Path
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL_PATH = Path(__file__).parent / "models" / "model.joblib"
package = joblib.load(MODEL_PATH)

model = package["model"]
FEATURES = package["features"]

categorical_features = package["categorical_features"]
numerical_features = package["numerical_features"]

cat_imputer = package["cat_imputer"]
num_imputer = package["num_imputer"]
encoder = package["encoder"]


app = FastAPI(title="Diabetes Prediction API")


class PatientData(BaseModel):
    gender: str
    age: float = Field(ge=0, le=120)
    hypertension: int = Field(ge=0, le=1)
    heart_disease: int = Field(ge=0, le=1)
    smoking_history: str
    bmi: float = Field(gt=0, le=150)
    HbA1c_level: float = Field(gt=0)
    blood_glucose_level: float = Field(gt=0)


@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/Prediction")
def predict(data: PatientData):
    input_data = pd.DataFrame(
        [data.model_dump()],
        columns=FEATURES
    )

    # Preprocess categorical features
    cat_data = cat_imputer.transform(
        input_data[categorical_features]
    )
    cat_data = encoder.transform(cat_data).toarray()

    # Preprocess numerical features
    num_data = num_imputer.transform(
        input_data[numerical_features]
    )

    # Combine features in the same order as training
    processed_data = np.hstack([cat_data, num_data])

    prediction = int(model.predict(processed_data)[0])
    probability = float(model.predict_proba(processed_data)[0][1])

    return {
        "prediction": prediction,
        "label": (
            "diabetes_class_1" if prediction == 1
            else "diabetes_class_0"
        ),
        "class_1_probability": round(probability, 4)
    }