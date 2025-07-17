from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's message in the conversation")
    session_id: Optional[str] = Field(None, description="Unique identifier for the conversation session")

class ChatResponse(BaseModel):
    response: str = Field(..., description="The chatbot's response")
    session_id: str = Field(..., description="Unique identifier for the conversation session")
    filter_decision: Optional[str] = Field(
        None,
        description="Guardrail filter decision: 'safe' or 'danger'",
    )
    filter_evaluation: Optional[str] = Field(
        None,
        description="Guardrail filter evaluation message when decision is 'danger'",
    )

class EvaluateResponseRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User's original prompt")
    response: str = Field(..., description="Model's generated response to evaluate")

class EvaluateResponseResponse(BaseModel):
    results: Dict[str, Any] = Field(
        ..., description="Mapping from dimension name to evaluation output"
    )