from pydantic import BaseModel

class UserFeatures(BaseModel):
    total_sessions: int
    total_adds: int
    # Add other features defined in your engineering step
    
class PredictionResponse(BaseModel):
    churn_probability: float
    is_at_risk: bool
