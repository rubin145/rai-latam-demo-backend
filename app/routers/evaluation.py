from fastapi import APIRouter, HTTPException
from typing import List
from ..models.schemas import (
    QueryRequest, QueryEvaluationResponse, 
    TestQuestion, TestQuestionRequest, TestQuestionResponse,
    BatchTestResponse, StatusResponse
)
from ..services.ai_refinery_service import AIRefineryService

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])

# Initialize service
ai_service = AIRefineryService()

@router.post("/query", response_model=QueryEvaluationResponse)
async def evaluate_query(request: QueryRequest):
    """Evaluate a single query for harm assessment"""
    try:
        result = await ai_service.evaluate_query(request.query)
        return QueryEvaluationResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Get service status"""
    status = ai_service.get_status()
    return StatusResponse(**status)

@router.get("/questions", response_model=List[TestQuestion])
async def get_test_questions():
    """Get all test questions"""
    try:
        questions = ai_service.get_all_test_questions()
        return questions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve questions: {str(e)}")

@router.post("/questions", response_model=TestQuestionResponse)
async def add_test_question(request: TestQuestionRequest):
    """Add a new test question"""
    try:
        success, message, question = ai_service.add_test_question(
            request.query, 
            request.expected_risk, 
            request.category, 
            request.description
        )
        
        return TestQuestionResponse(
            success=success,
            message=message,
            question=question
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add question: {str(e)}")

@router.post("/batch-test", response_model=BatchTestResponse)
async def run_batch_test():
    """Run batch test on all questions"""
    try:
        result = await ai_service.run_batch_test()
        return BatchTestResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run batch test: {str(e)}") 