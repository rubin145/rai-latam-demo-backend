from fastapi import APIRouter, HTTPException, Request, Response
from ..models.schemas import (
    ChatRequest, ChatResponse,
    EvaluateResponseRequest, EvaluateResponseResponse,
    ChatbotInfo, ChatbotsListResponse,
    HumanFeedbackRequest, DatasetEvaluationRequest
)
from ..services.chat import ChatService
from ..services.evaluators import LLMEvaluator
from ..services.langsmith_client import LangSmithClient
import uuid
import os
import glob
import yaml

router = APIRouter(prefix="/api")

@router.get("/chatbots", response_model=ChatbotsListResponse)
async def list_chatbots() -> ChatbotsListResponse:
    """List all available chatbots/projects"""
    chatbots = []
    
    # Scan chatbot config files
    config_files = glob.glob("configs/chatbots/*.yaml")
    
    for config_file in config_files:
        # Extract chatbot_id from filename
        filename = os.path.basename(config_file)
        chatbot_id = filename.replace('.yaml', '')
        
        try:
            # Load config to determine if it has guardrails
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            has_guardrails = bool(config.get("input_filters"))
            
            # Create human-readable name and description
            if "safe" in chatbot_id:
                name = chatbot_id.replace("_safe", "").title() + " (Safe)"
                description = f"AI assistant with safety guardrails enabled"
            elif "unsafe" in chatbot_id:
                name = chatbot_id.replace("_unsafe", "").title() + " (Unsafe)"
                description = f"AI assistant without safety filters"
            else:
                name = chatbot_id.title()
                description = f"AI assistant"
            
            chatbots.append(ChatbotInfo(
                id=chatbot_id,
                name=name,
                description=description,
                has_guardrails=has_guardrails
            ))
            
        except Exception as e:
            print(f"⚠️ Error loading config {config_file}: {e}")
            continue
    
    return ChatbotsListResponse(chatbots=chatbots)

@router.post("/chatbots/{chatbot_id}/chat", response_model=ChatResponse)
async def chat_chatbot_endpoint(
    chatbot_id: str,
    chat_request: ChatRequest,
    response: Response,
) -> ChatResponse:
    """Main chat endpoint for multi-chatbot support"""
    # Use chatbot_id directly as config file name
    config_path = f"configs/chatbots/{chatbot_id}.yaml"

    # Validate that the configuration file for the chatbot exists
    if not os.path.exists(config_path):
        raise HTTPException(
            status_code=404,
            detail=f"Chatbot configuration not found for id '{chatbot_id}'."
        )

    try:
        # Create service dynamically and determine if it has guardrails
        chat_service = ChatService(config_path)
        has_guardrails = bool(chat_service.config.get("input_filters"))
        
        filter_triggered = False
        filter_evaluation = None
        actual_response = None
        run_id = None
        
        # Apply input filters if chatbot has guardrails
        if has_guardrails:
            filter_decision, filter_eval, template_response = await chat_service.apply_input_filters(chat_request.query)
            
            if filter_decision == "danger":
                print(f"🚨 [GUARDRAIL] {chatbot_id} - evaluation={filter_eval!r} 🚨")
                filter_triggered = True
                filter_evaluation = filter_eval
                actual_response = template_response
            else:
                # Process chat normally when filters pass
                print(f"👤 User ({chatbot_id}) -> {chat_request.query!r} (session_id={chat_request.session_id!r})")
                actual_response, session_id, run_id = await chat_service.handle_chat(
                    chat_request.query, chat_request.session_id
                )
                print(f"🤖 Agent ({chatbot_id}) -> {actual_response!r} (session_id={session_id!r}) (run_id={run_id})")
        else:
            # No guardrails - process chat directly
            print(f"👤 User ({chatbot_id}) -> {chat_request.query!r} (session_id={chat_request.session_id!r})")
            actual_response, session_id, run_id = await chat_service.handle_chat(
                chat_request.query, chat_request.session_id
            )
            print(f"🤖 Agent ({chatbot_id}) -> {actual_response!r} (session_id={session_id!r}) (run_id={run_id})")
        
        # Handle session ID
        if not chat_request.session_id:
            session_id = session_id if 'session_id' in locals() else str(uuid.uuid4())
            response.set_cookie(
                key="session_id",
                value=session_id,
                httponly=True,
            )
        else:
            session_id = chat_request.session_id
        
        # Return response with new schema
        return ChatResponse(
            response=actual_response,
            session_id=session_id,
            run_id=run_id,
            guardrails_active=has_guardrails,
            filter_triggered=filter_triggered,
            filter_evaluation=filter_evaluation,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate", response_model=EvaluateResponseResponse)
async def evaluate_response_endpoint(
    payload: EvaluateResponseRequest
) -> EvaluateResponseResponse:
    """Evaluate a model's response across configured dimensions."""
    evaluator = LLMEvaluator("configs/evaluators/llm_evaluators.yaml")
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
    evaluator_service = LangSmithClient()
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
    evaluator_service = LangSmithClient()
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
    evaluator_service = LangSmithClient()

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
            results = await evaluator_service.evaluator_manager.evaluate_response(
                payload.prompt, payload.response
            )
            return {
                "evaluation_results": results,
                "langsmith_feedback_added": False
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/langsmith/evaluate_trace_readonly/{run_id}")
async def evaluate_trace_readonly(
    run_id: str,
    evaluator_names: list[str] = None
):
    """Evaluate a LangSmith trace by run ID WITHOUT adding feedback to the trace."""
    evaluator_service = LangSmithClient()
    try:
        results = await evaluator_service.evaluate_trace_readonly(run_id, evaluator_names)
        return {
            "run_id": run_id,
            "evaluation_results": results,
            "evaluators_used": evaluator_names or evaluator_service.evaluator_names,
            "feedback_added": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/langsmith/evaluate_dataset/{dataset_id}")
async def evaluate_dataset(
    dataset_id: str,
    request: DatasetEvaluationRequest
):
    """Evaluate all examples in a LangSmith dataset."""
    evaluator_service = LangSmithClient()
    try:
        results = await evaluator_service.evaluate_dataset(
            dataset_id, 
            request.evaluator_names, 
            request.add_feedback
        )
        return {
            "dataset_evaluation": results,
            "evaluators_used": request.evaluator_names or evaluator_service.evaluator_names,
            "feedback_added": request.add_feedback
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/langsmith/human_feedback/{run_id}")
async def add_human_feedback(
    run_id: str,
    feedback: HumanFeedbackRequest
):
    """Add human feedback to a LangSmith trace."""
    evaluator_service = LangSmithClient()
    try:
        result = await evaluator_service.record_human_feedback(
            run_id=run_id,
            feedback_type=feedback.feedback_type,
            value=feedback.value,
            comment=feedback.comment
        )
        return {
            "run_id": run_id,
            "feedback_added": True,
            "feedback_type": feedback.feedback_type,
            "value": feedback.value,
            "comment": feedback.comment,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))