from fastapi import APIRouter, HTTPException, Request, Response, Query
from ..models.schemas import (
    ChatRequest, ChatResponse,
    EvaluateResponseRequest, EvaluateResponseResponse,
)
from ..services.langchain_chat import LangChainChatService
from ..services.response_evaluator import ResponseEvaluatorService
import json
import uuid
import os

router = APIRouter(prefix="/api")

@router.post("/chatbots/{chatbot_id}/chat", response_model=ChatResponse)
async def chat_chatbot_endpoint(
    chatbot_id: str,
    chat_request: ChatRequest,
    response: Response,
    use_guardrails: bool = Query(default=True, description="Whether to apply input filters/guardrails")
) -> ChatResponse:
    """Main chat endpoint for multi-chatbot support"""
    try:
        # Determine config file based on chatbot and guardrails setting
        config_suffix = "safe" if use_guardrails else "unsafe"
        config_path = f"configs/chatbots/{chatbot_id}_{config_suffix}.yaml"
        
        # Create service dynamically
        chat_service = LangChainChatService(config_path)
        
        # Apply input filters if guardrails are enabled
        if use_guardrails:
            filter_decision, filter_evaluation, template_response = await chat_service.apply_input_filters(chat_request.query)
            
            if filter_decision == "danger":
                print(f"🚨 [GUARDRAIL] {chatbot_id} - evaluation={filter_evaluation!r} 🚨")
                sid = chat_request.session_id or str(uuid.uuid4())
                if not chat_request.session_id:
                    response.set_cookie(key="session_id", value=sid, httponly=True)
                return ChatResponse(
                    response=template_response,
                    session_id=sid,
                    filter_decision=filter_decision,
                    filter_evaluation=filter_evaluation,
                )
        
        # Process chat normally
        print(f"👤 User ({chatbot_id}) -> {chat_request.query!r} (session_id={chat_request.session_id!r})")
        message, session_id = await chat_service.handle_chat(
            chat_request.query, chat_request.session_id
        )
        print(f"🤖 Agent ({chatbot_id}) -> {message!r} (session_id={session_id!r})")
        
        # Set session cookie if needed
        if not chat_request.session_id:
            response.set_cookie(
                key="session_id",
                value=session_id,
                httponly=True,
            )
        
        # Return response with filter information
        return ChatResponse(
            response=message,
            session_id=session_id,
            filter_decision="safe" if use_guardrails else None,
            filter_evaluation="" if use_guardrails else None,
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Chatbot '{chatbot_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate_response", response_model=EvaluateResponseResponse)
async def evaluate_response_endpoint(
    request: Request, payload: EvaluateResponseRequest
) -> EvaluateResponseResponse:
    """Evaluate a model's response across configured dimensions."""
    evaluator = ResponseEvaluatorService("configs/evaluators/response_evaluator.yaml")
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
    """Legacy endpoint for standard chat - routes to banking_unsafe"""
    return await chat_chatbot_endpoint(
        chatbot_id="banking",
        chat_request=chat_request,
        response=response,
        use_guardrails=False
    )

@router.post("/chat-guardrails", response_model=ChatResponse)
async def chat_guardrails_endpoint(
    request: Request,
    chat_request: ChatRequest,
    response: Response,
) -> ChatResponse:
    """Legacy endpoint for chat with guardrails - routes to banking_safe"""
    return await chat_chatbot_endpoint(
        chatbot_id="banking",
        chat_request=chat_request,
        response=response,
        use_guardrails=True
    )