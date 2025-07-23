from fastapi import APIRouter, HTTPException, Request, Response, Query, BackgroundTasks
from ..models.schemas import (
    ChatRequest, ChatResponse,
    EvaluateResponseRequest, EvaluateResponseResponse,
    HumanFeedbackRequest, HumanFeedbackResponse,
)
from ..services.langchain_chat import LangChainChatService
from ..services.response_evaluator import ResponseEvaluatorService
from ..dependencies import get_evaluation_service
import json
import uuid
import os

router = APIRouter(prefix="/api")

@router.post("/chatbots/{chatbot_id}/chat", response_model=ChatResponse)
async def chat_chatbot_endpoint(
    chatbot_id: str,
    chat_request: ChatRequest,
    response: Response,
    background_tasks: BackgroundTasks,
    use_guardrails: bool = Query(default=True, description="Whether to apply input filters/guardrails")
) -> ChatResponse:
    """Main chat endpoint for multi-chatbot support with integrated evaluation"""
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
        # Create chat service
        chat_service = LangChainChatService(config_path)
        
        # Get evaluation service for real-time feedback
        eval_service = get_evaluation_service(chatbot_id)
        
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
        
        # Get real-time evaluation callbacks
        callbacks = eval_service.get_realtime_callbacks()
        
        # Process chat with evaluation callbacks
        print(f"👤 User ({chatbot_id}) -> {chat_request.query!r} (session_id={chat_request.session_id!r})")
        
        # Modified to include callbacks for real-time evaluation
        message, session_id = await chat_service.handle_chat(
            chat_request.query, 
            chat_request.session_id,
            callbacks=callbacks  # Pass evaluation callbacks
        )
        
        print(f"🤖 Agent ({chatbot_id}) -> {message!r} (session_id={session_id!r})")
        print(f"📊 [EVALUATION] Real-time feedback active for {chatbot_id}")
        
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

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: Request,
    chat_request: ChatRequest,
    response: Response,
    background_tasks: BackgroundTasks,
) -> ChatResponse:
    """Legacy endpoint for standard chat - routes to banking_unsafe"""
    return await chat_chatbot_endpoint(
        chatbot_id="banking",
        chat_request=chat_request,
        response=response,
        background_tasks=background_tasks,
        use_guardrails=False
    )

@router.post("/chat-guardrails", response_model=ChatResponse)
async def chat_guardrails_endpoint(
    request: Request,
    chat_request: ChatRequest,
    response: Response,
    background_tasks: BackgroundTasks,
) -> ChatResponse:
    """Legacy endpoint for chat with guardrails - routes to banking_safe"""
    return await chat_chatbot_endpoint(
        chatbot_id="banking",
        chat_request=chat_request,
        response=response,
        background_tasks=background_tasks,
        use_guardrails=True
    )

# New Integrated Evaluation Endpoints

@router.post("/feedback", response_model=HumanFeedbackResponse)
async def submit_human_feedback(
    feedback: HumanFeedbackRequest,
    project_name: str = Query(default="banking", description="Project name for evaluation service")
) -> HumanFeedbackResponse:
    """Submit human feedback (thumbs up/down, ratings) for a conversation turn"""
    try:
        eval_service = get_evaluation_service(project_name)
        result = await eval_service.record_human_feedback(
            run_id=feedback.run_id,
            feedback_type=feedback.feedback_type,
            value=feedback.value,
            comment=feedback.comment
        )
        return HumanFeedbackResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate_response_integrated", response_model=EvaluateResponseResponse)
async def evaluate_response_integrated_endpoint(
    request: EvaluateResponseRequest,
    project_name: str = Query(default="banking", description="Project name for evaluation service")
) -> EvaluateResponseResponse:
    """On-demand response evaluation using the integrated evaluation service"""
    try:
        eval_service = get_evaluation_service(project_name)
        results = await eval_service.evaluate_on_demand(
            prompt=request.prompt,
            response=request.response
        )
        return EvaluateResponseResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))