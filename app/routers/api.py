from fastapi import APIRouter, HTTPException, Request, Response
from typing import List
from ..models.schemas import (
    QueryRequest, QueryEvaluationResponse, 
    TestQuestion, TestQuestionRequest, TestQuestionResponse,
    BatchTestResponse, StatusResponse, ChatRequest, ChatResponse
)
from ..services.ai_refinery_service import AIRefineryService
from ..services.chat import ChatService

router = APIRouter(prefix="/api")

@router.post("/evaluation/query", response_model=QueryEvaluationResponse)
async def evaluate_query(request: Request, query_request: QueryRequest):
    """Evaluate a single query for harm assessment"""
    client = request.app.state.distiller_client
    ai_service = AIRefineryService(client)
    try:
        result = await ai_service.evaluate_query(query_request.query)
        return QueryEvaluationResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/evaluation/status", response_model=StatusResponse)
async def get_status(request: Request):
    """Get service status"""
    client = request.app.state.distiller_client
    ai_service = AIRefineryService(client)
    status = ai_service.get_status() # This method needs to be checked if it uses the client
    return StatusResponse(**status)

@router.get("/evaluation/questions", response_model=List[TestQuestion])
async def get_test_questions(request: Request):
    """Get all test questions"""
    client = request.app.state.distiller_client
    ai_service = AIRefineryService(client)
    try:
        questions = ai_service.get_all_test_questions()
        return questions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve questions: {str(e)}")

@router.post("/evaluation/questions", response_model=TestQuestionResponse)
async def add_test_question(request: Request, test_question_request: TestQuestionRequest):
    """Add a new test question"""
    client = request.app.state.distiller_client
    ai_service = AIRefineryService(client)
    try:
        success, message, question = ai_service.add_test_question(
            test_question_request.query, 
            test_question_request.expected_risk, 
            test_question_request.category, 
            test_question_request.description
        )
        
        return TestQuestionResponse(
            success=success,
            message=message,
            question=question
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add question: {str(e)}")

@router.post("/evaluation/batch-test", response_model=BatchTestResponse)
async def run_batch_test(request: Request):
    """Run batch test on all questions"""
    client = request.app.state.distiller_client
    ai_service = AIRefineryService(client)
    try:
        result = await ai_service.run_batch_test()
        return BatchTestResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run batch test: {str(e)}")

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: Request,
    chat_request: ChatRequest,
    response: Response,
) -> ChatResponse:
    """Endpoint for standard chat"""
    client = request.app.state.distiller_client
    chat_service = ChatService(client, "chat_project")
    try:
        message, session_id = await chat_service.handle_chat(
            chat_request.query, chat_request.session_id
        )
        # On first turn, set a session_id cookie so the front end can reuse it automatically
        if not chat_request.session_id:
            # Persist session_id in a secure HTTP-only cookie
            response.set_cookie(
                key="session_id",
                value=session_id,
                httponly=True,
            )
        return ChatResponse(response=message, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat-guardrails", response_model=ChatResponse)
async def chat_guardrails_endpoint(
    request: Request,
    chat_request: ChatRequest,
    response: Response,
) -> ChatResponse:
    """Endpoint for chat with guardrails"""
    client = request.app.state.distiller_client
    chat_service = ChatService(client, "chat_guardrails_project")
    try:
        message, session_id = await chat_service.handle_chat(
            chat_request.query, chat_request.session_id
        )
        if not chat_request.session_id:
            # Persist session_id in a secure HTTP-only cookie
            response.set_cookie(
                key="session_id",
                value=session_id,
                httponly=True,
            )
        return ChatResponse(response=message, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))