from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's message in the conversation")
    session_id: Optional[str] = Field(None, description="Unique identifier for the conversation session")

class ChatResponse(BaseModel):
    response: str = Field(..., description="The chatbot's response")
    session_id: str = Field(..., description="Unique identifier for the conversation session")
    run_id: Optional[str] = Field(None, description="LangSmith run ID for this interaction (for feedback)")
    guardrails_active: bool = Field(..., description="Whether this chatbot has guardrails enabled")
    filter_triggered: bool = Field(False, description="Whether a guardrail filter was triggered")
    filter_evaluation: Optional[str] = Field(
        None,
        description="Guardrail filter evaluation message when filter was triggered",
    )

class EvaluateResponseRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User's original prompt")
    response: str = Field(..., description="Model's generated response to evaluate")

class EvaluateResponseResponse(BaseModel):
    results: Dict[str, Any] = Field(
        ..., description="Mapping from dimension name to evaluation output"
    )

class ChatbotInfo(BaseModel):
    id: str = Field(..., description="Unique identifier for the chatbot")
    name: str = Field(..., description="Human-readable name for the chatbot")
    description: str = Field(..., description="Description of the chatbot's purpose")
    has_guardrails: bool = Field(..., description="Whether this chatbot has safety guardrails enabled")

class ChatbotsListResponse(BaseModel):
    chatbots: List[ChatbotInfo] = Field(..., description="List of available chatbots")

class HumanFeedbackRequest(BaseModel):
    feedback_type: str = Field(..., description="Type of feedback: 'thumbs', 'rating', etc.")
    value: str = Field(..., description="Feedback value: 'up'/'down' for thumbs, '1-5' for rating")
    comment: Optional[str] = Field(None, description="Optional comment with the feedback")

class DatasetEvaluationRequest(BaseModel):
    evaluator_names: Optional[List[str]] = Field(None, description="List of evaluator names to use (default: all)")
    add_feedback: bool = Field(True, description="Whether to add feedback to traces in the dataset")