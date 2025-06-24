from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Query to evaluate for harm")

class QueryEvaluationResponse(BaseModel):
    assessment: str
    dimensions: Dict[str, Any]
    clean_response: str
    full_response: str
    raw_agent_output: Optional[str] = ""

class TestQuestion(BaseModel):
    id: Optional[int] = None
    query: str = Field(..., min_length=1)
    expected_risk: str
    category: str
    description: str
    date_added: Optional[str] = None

class TestQuestionRequest(BaseModel):
    query: str = Field(..., min_length=1)
    expected_risk: str = Field(..., pattern="^(Safe|Moderate|High)$")
    category: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)

class TestQuestionResponse(BaseModel):
    success: bool
    message: str
    question: Optional[TestQuestion] = None

class TestQuestionsDatabase(BaseModel):
    safe_queries: List[TestQuestion]
    mild_risk_queries: List[TestQuestion]
    high_risk_queries: List[TestQuestion]
    edge_cases: List[TestQuestion]
    metadata: Dict[str, Any]

class BatchTestResult(BaseModel):
    question_id: int
    query: str
    expected_risk: str
    category: str
    assessment: str
    dimensions: Dict[str, str]
    match_found: bool

class BatchTestResponse(BaseModel):
    results: List[BatchTestResult]
    total_questions: int
    matches_found: int
    accuracy_percentage: float

class StatusResponse(BaseModel):
    status: str
    message: str
    timestamp: str

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