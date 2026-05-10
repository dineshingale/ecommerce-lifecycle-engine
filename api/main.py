from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# Internal schema definition for instant compatibility
class UserFeatures(BaseModel):
    total_sessions: int
    add_to_cart_rate: float  # = total_adds / (total_sessions + 1)

class PredictionResponse(BaseModel):
    churn_probability: float
    is_at_risk: bool

app = FastAPI(title="E-commerce Churn Predictor")

# Load the saved model pipeline
model = joblib.load("models/churn_model.pkl")

@app.get("/")
def home():
    return {
        "message": "Churn Prediction API is Online",
        "project": "SmartEcommerceForecaster"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(data: UserFeatures):
    # Convert incoming JSON to Dataframe for the pipeline
    input_df = pd.DataFrame([data.model_dump()])
    
    # Get probability from XGBoost
    # [0][1] gets the probability for the positive class (Churn)
    prob = model.predict_proba(input_df)[0][1]
    
    return {
        "churn_probability": round(float(prob), 4),
        "is_at_risk": bool(prob > 0.7)  # Threshold for business action
    }
