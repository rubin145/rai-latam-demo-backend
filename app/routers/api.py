from fastapi import APIRouter, HTTPException, Request, Response
from typing import List
from ..models.schemas import (
    QueryRequest, QueryEvaluationResponse,
    TestQuestion, TestQuestionRequest, TestQuestionResponse,
    BatchTestResponse, StatusResponse, ChatRequest, ChatResponse,
    EvaluateResponseRequest, EvaluateResponseResponse,
)
from ..services.harm_evaluator_service import HarmEvaluatorService
from ..services.air_chat import AIRChatService
from ..services.groq_chat import GroqChatService
from ..services.response_evaluator_service import ResponseEvaluatorService
import json
import uuid
from groq import Groq
from ..utils.config_loader import load_yaml
import os

router = APIRouter(prefix="/api")

@router.post("/evaluation/query", response_model=QueryEvaluationResponse)
async def evaluate_query(request: Request, query_request: QueryRequest):
    """Evaluate a single query for harm assessment"""
    client = request.app.state.distiller_client
    ai_service = HarmEvaluatorService(client)
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
    ai_service = HarmEvaluatorService(client)
    status = ai_service.get_status()
    return StatusResponse(**status)

@router.get("/evaluation/questions", response_model=List[TestQuestion])
async def get_test_questions(request: Request):
    """Get all test questions"""
    client = request.app.state.distiller_client
    ai_service = HarmEvaluatorService(client)
    try:
        return ai_service.get_all_test_questions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve questions: {e}")

@router.post("/evaluation/questions", response_model=TestQuestionResponse)
async def add_test_question(request: Request, test_question_request: TestQuestionRequest):
    """Add a new test question"""
    client = request.app.state.distiller_client
    ai_service = HarmEvaluatorService(client)
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
        raise HTTPException(status_code=500, detail=f"Failed to add question: {e}")

@router.post("/evaluation/batch-test", response_model=BatchTestResponse)
async def run_batch_test(request: Request):
    """Run batch test on all questions"""
    client = request.app.state.distiller_client
    ai_service = HarmEvaluatorService(client)
    try:
        result = await ai_service.run_batch_test()
        return BatchTestResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run batch test: {e}")

@router.post("/evaluate_response", response_model=EvaluateResponseResponse)
async def evaluate_response_endpoint(
    request: Request, payload: EvaluateResponseRequest
) -> EvaluateResponseResponse:
    """Evaluate a model's response across configured dimensions."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500, detail="GROQ_API_KEY not configured"
        )
    evaluator = ResponseEvaluatorService(api_key, "groq_response_evaluator_config.yaml")
    try:
        results = await evaluator.evaluate_response(payload.prompt, payload.response)
        return EvaluateResponseResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: Request,
    chat_request: ChatRequest,
    response: Response,
) -> ChatResponse:
    """Endpoint for standard chat"""
    svc = request.app.state.base_chat_service
    if not svc:
        raise HTTPException(status_code=503, detail="Chat service unavailable")
    try:
        print(f"👤 User -> {chat_request.query!r} (session_id={chat_request.session_id!r})")
        message, session_id = await svc.handle_chat(
            chat_request.query, chat_request.session_id
        )
        print(f"🤖 Agent -> {message!r} (session_id={session_id!r})")
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
    svc = request.app.state.guardrails_chat_service
    if not svc:
        raise HTTPException(status_code=503, detail="Chat service unavailable")
    try:
        # Apply Groq-based input filters only when using the Groq guardrails service
        if isinstance(svc, GroqChatService):
            cfg = load_yaml("groq_chat_rai_config.yaml")
            for flt in cfg.get("input_filters", []):
                client_filter = Groq()
                msgs = []
                if flt.get("system_prompt"):
                    msgs.append({"role": "system", "content": flt["system_prompt"]})
                msgs.append({"role": "user", "content": chat_request.query.strip()})
                resp = client_filter.chat.completions.create(
                    model=flt.get("model"),
                    messages=msgs,
                    **flt.get("inference", {})
                )
                content = ""
                if hasattr(resp, "choices") and resp.choices:
                    content = resp.choices[0].message.content
                try:
                    outcome = json.loads(content)
                except Exception:
                    continue
                if outcome.get("decision") == "danger":
                    print(f"🚨 [GUARDRAIL:{flt.get('name')}] evaluation={outcome.get('evaluation')!r} 🚨")
                    sid = chat_request.session_id or str(uuid.uuid4())
                    if not chat_request.session_id:
                        response.set_cookie(key="session_id", value=sid, httponly=True)
                    return ChatResponse(
                        response=flt.get("template_response", "Sorry, I can’t help with that."),
                        session_id=sid,
                        filter_decision=outcome.get("decision"),
                        filter_evaluation=f"{flt.get('name')}: {outcome.get('evaluation')}",
                    )
        # Passed all filters or using AIR guardrails, continue with main agent
        print(f"👤 User -> {chat_request.query!r} (session_id={chat_request.session_id!r})")
        message, session_id = await svc.handle_chat(
            chat_request.query, chat_request.session_id
        )
        print(f"🤖 Agent [guardrails] -> {message!r} (session_id={session_id!r})")
        if not chat_request.session_id:
            # Persist session_id in a secure HTTP-only cookie
            response.set_cookie(
                key="session_id",
                value=session_id,
                httponly=True,
            )
        # For Groq guardrails we return filter flags; AIR guardrails will skip filters
        if isinstance(svc, GroqChatService):
            return ChatResponse(
                response=message,
                session_id=session_id,
                filter_decision="safe",
                filter_evaluation="",
            )
        return ChatResponse(response=message, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))