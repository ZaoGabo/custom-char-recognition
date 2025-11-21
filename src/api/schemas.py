from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class PredictionRequest(BaseModel):
    image: List[float] = Field(
        ..., 
        description="Flattened image array. Values should be normalized [0, 1].",
        example=[0.0] * 784
    )
    width: Optional[int] = Field(None, description="Image width (required for non-square images)")
    height: Optional[int] = Field(None, description="Image height (required for non-square images)")

class Top5Prediction(BaseModel):
    character: str = Field(..., description="Predicted character")
    probability: float = Field(..., description="Probability score")

class PredictionResponse(BaseModel):
    character: str = Field(..., description="Predicted character")
    confidence: float = Field(..., description="Confidence score (0-1)")
    top5: List[Top5Prediction] = Field(..., description="Top 5 predictions with probabilities")

class PredictionTextResponse(BaseModel):
    text: str = Field(..., description="Predicted text string")
    segments: List[PredictionResponse] = Field(..., description="Detailed prediction for each character segment")
