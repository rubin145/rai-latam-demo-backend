from fastapi import APIRouter, HTTPException, Request, Response, Query
from ..models.schemas import (
    ChatRequest, ChatResponse,
    EvaluateResponseRequest, EvaluateResponseResponse,
)
from ..services.langchain_chat import LangChainChatService
from ..services.response_evaluator import ResponseEvaluatorService
from ..services.langsmith_evaluator import LangSmithEvaluatorService
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
    # Determine config file based on chatbot and guardrails setting
    config_suffix = "safe" if use_guardrails else "unsafe"
    config_path = f"configs/chatbots/{chatbot_id}_{config_suffix}.yaml"

    # Validate that the configuration file for the chatbot exists
    if not os.path.exists(config_path):
        raise HTTPException(
            status_code=404,
            detail=f"Chatbot configuration not found for id '{chatbot_id}' with guardrails={use_guardrails}."
        )

    try:
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate_response", response_model=EvaluateResponseResponse)
async def evaluate_response_endpoint(
    request: Request, payload: EvaluateResponseRequest
) -> EvaluateResponseResponse:
    """Evaluate a model's response across configured dimensions."""
    evaluator = ResponseEvaluatorService("configs/evaluators/llm_evaluators.yaml")
    try:
        results = await evaluator.evaluate_response(payload.prompt, payload.response)
        return EvaluateResponseResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/langsmith/evaluate_trace/{run_id}")
async def evaluate_trace_endpoint(
    run_id: str,
    evaluator_names: list[str] = None
):
    """Evaluate a specific LangSmith trace by run ID using LLM-as-a-judge evaluators."""
    evaluator_service = LangSmithEvaluatorService()
    try:
        results = await evaluator_service.evaluate_single_trace(run_id, evaluator_names)
        return {
            "run_id": run_id,
            "evaluation_results": results,
            "evaluators_used": evaluator_names or evaluator_service.evaluator_names
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/langsmith/evaluators")
async def list_langsmith_evaluators():
    """List all available LangSmith LLM-as-a-judge evaluators."""
    evaluator_service = LangSmithEvaluatorService()
    try:
        evaluators = evaluator_service.get_available_evaluators()
        return {
            "evaluators": evaluators,
            "count": len(evaluators)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/langsmith/evaluate_response")
async def evaluate_response_with_langsmith_feedback(
    request: Request,
    payload: EvaluateResponseRequest
):
    """Evaluate a response and optionally add feedback to a LangSmith trace."""
    evaluator_service = LangSmithEvaluatorService()
    
    # Get optional run_id from headers or query params
    run_id = request.headers.get("x-langsmith-run-id") or request.query_params.get("run_id")
    
    try:
        if run_id:
            # Evaluate and add feedback to trace
            results = await evaluator_service.evaluate_and_add_feedback(
                run_id=run_id,
                prompt=payload.prompt,
                response=payload.response
            )
            return {
                "evaluation_results": results,
                "langsmith_feedback_added": True,
                "run_id": run_id
            }
        else:
            # Just evaluate without adding to trace
            results = await evaluator_service.response_evaluator.evaluate_response(
                payload.prompt, payload.response
            )
            return {
                "evaluation_results": results,
                "langsmith_feedback_added": False
            }
            
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