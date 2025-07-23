from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Literal

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

# New schemas for human feedback
class HumanFeedbackRequest(BaseModel):
    run_id: str = Field(..., description="LangSmith run ID for the conversation turn")
    feedback_type: Literal["thumbs", "score"] = Field(..., description="Type of feedback")
    value: Any = Field(..., description="Feedback value ('up'/'down' for thumbs, 1-5 for score)")
    comment: Optional[str] = Field(None, description="Optional comment with the feedback")

class HumanFeedbackResponse(BaseModel):
    status: str = Field(..., description="Status of feedback recording")
    message: Optional[str] = Field(None, description="Additional status message")
    run_id: Optional[str] = Field(None, description="The run ID that was updated")

