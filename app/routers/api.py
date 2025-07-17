from fastapi import APIRouter, HTTPException, Request, Response
from ..models.schemas import (
    ChatRequest, ChatResponse,
    EvaluateResponseRequest, EvaluateResponseResponse,
)
from ..services.langchain_chat import LangChainChatService
from ..services.response_evaluator_service import ResponseEvaluatorService
import json
import uuid
import os

router = APIRouter(prefix="/api")

@router.post("/evaluate_response", response_model=EvaluateResponseResponse)
async def evaluate_response_endpoint(
    request: Request, payload: EvaluateResponseRequest
) -> EvaluateResponseResponse:
    """Evaluate a model's response across configured dimensions."""
    evaluator = ResponseEvaluatorService("response_evaluator_config.yaml")
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
        # Apply input filters using LangChain
        if isinstance(svc, LangChainChatService):
            filter_decision, filter_evaluation, template_response = await svc.apply_input_filters(chat_request.query)
            
            if filter_decision == "danger":
                print(f"🚨 [GUARDRAIL] evaluation={filter_evaluation!r} 🚨")
                sid = chat_request.session_id or str(uuid.uuid4())
                if not chat_request.session_id:
                    response.set_cookie(key="session_id", value=sid, httponly=True)
                return ChatResponse(
                    response=template_response,
                    session_id=sid,
                    filter_decision=filter_decision,
                    filter_evaluation=filter_evaluation,
                )
        
        # Passed all filters, continue with main agent
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
        # For LangChain services we return filter flags
        if isinstance(svc, LangChainChatService):
            return ChatResponse(
                response=message,
                session_id=session_id,
                filter_decision="safe",
                filter_evaluation="",
            )
        return ChatResponse(response=message, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))