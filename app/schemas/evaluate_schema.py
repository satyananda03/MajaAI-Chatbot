from pydantic import BaseModel
from typing import List
class EvaluateRequest(BaseModel):
    question: str

class EvaluateResponse(BaseModel):
    # answer: str
    context: List[str]
    latency: float