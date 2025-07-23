import os
import random
from typing import Any, Dict, List, Optional
from langsmith.evaluation.evaluator import run_evaluator, EvaluationResult
from langsmith.schemas import Run
# from langchain_core.tracers.evaluation import EvaluatorCallbackHandler  # Not needed with optimized approach
from langchain_groq import ChatGroq
from langchain.evaluation.criteria import CriteriaEvalChain
from langchain.evaluation.scoring import ScoreStringEvalChain
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
from langsmith import Client

from ..utils.config_loader import load_yaml


# Lightweight evaluators using @run_evaluator decorator

@run_evaluator
def response_length_evaluator(run: Run, example=None) -> EvaluationResult:
    """Evaluate response length (10-2000 characters optimal)"""
    response = run.outputs.get("response", "") if run.outputs else ""
    length = len(response)
    score = 1.0 if 10 <= length <= 2000 else 0.3
    comment = f"Response length: {length} characters"
    if length < 10:
        comment += " (too short)"
    elif length > 2000:
        comment += " (too long)"
    else:
        comment += " (optimal)"
        
    return EvaluationResult(
        key="response_length",
        score=score,
        comment=comment
    )

@run_evaluator
def language_detection_evaluator(run: Run, example=None) -> EvaluationResult:
    """Basic language detection using character patterns"""
    response = run.outputs.get("response", "") if run.outputs else ""
    expected_language = "pt"
    
    # Simple Portuguese indicators
    pt_indicators = ['ã', 'ç', 'ção', 'ões', 'ão', 'em', 'de', 'do', 'da', 'para', 'com']
    
    if expected_language == "pt":
        pt_count = sum(1 for indicator in pt_indicators if indicator in response.lower())
        score = min(pt_count / 3, 1.0)  # Normalize to 0-1
    else:
        score = 1.0  # Default pass for other languages
        
    return EvaluationResult(
        key="language_detection",
        score=score,
        comment=f"Language detection score: {score:.2f} for {expected_language}"
    )

@run_evaluator
def content_safety_evaluator(run: Run, example=None) -> EvaluationResult:
    """Basic content safety using regex patterns"""
    import re
    
    response = run.outputs.get("response", "") if run.outputs else ""
    
    # Patterns for potentially sensitive content
    patterns = {
        'url': r'https?://[^\s]+',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{10,11}\b',  # Basic phone number pattern
    }
    
    issues = []
    for pattern_name, pattern in patterns.items():
        if re.search(pattern, response):
            issues.append(pattern_name)
    
    score = 1.0 if not issues else 0.5
    comment = f"Content safety check: {issues if issues else 'clean'}"
    
    return EvaluationResult(
        key="content_safety",
        score=score,
        comment=comment
    )


# Sampling evaluator factory functions

def create_sampled_evaluator(base_evaluator, sampling_rate: float = 1.0):
    """Create a sampled version of an evaluator"""
    @run_evaluator
    def sampled_evaluator(run: Run, example=None) -> EvaluationResult:
        # Apply sampling
        if random.random() > sampling_rate:
            return EvaluationResult(
                key=f"{base_evaluator.__name__}_sampled",
                score=None,
                comment="Skipped due to sampling"
            )
        
        # Call the base evaluator
        return base_evaluator(run, example)
    
    return sampled_evaluator


def create_sync_sampled_evaluator(evaluator_name: str, evaluator, sampling_rate: float = 0.1):
    """Create a sync sampled LLM evaluator that actually works"""
    @run_evaluator
    def sync_sampled_evaluator(run: Run, example=None) -> EvaluationResult:
        # Apply sampling
        if random.random() > sampling_rate:
            return EvaluationResult(
                key=evaluator_name,
                score=None,
                comment="Skipped due to sampling"
            )
        
        try:
            # Extract data from run
            inputs = run.inputs or {}
            outputs = run.outputs or {}
            
            prompt = inputs.get("input", inputs.get("query", ""))
            response = outputs.get("response", outputs.get("output", ""))
            
            if not prompt or not response:
                return EvaluationResult(
                    key=evaluator_name,
                    score=0.0,
                    comment="Missing prompt or response data"
                )
            
            # Run SYNC evaluation (simple and works!)
            result = evaluator.evaluate_strings(
                prediction=response,
                input=prompt
            )
            
            # Parse result based on evaluator type
            if "score" in result:
                score = result["score"]
                comment = result.get("reasoning", result.get("value", ""))
            else:
                score = 1.0 if result.get("value") == "Y" else 0.0
                comment = result.get("reasoning", "")
            
            return EvaluationResult(
                key=evaluator_name,
                score=score,
                comment=comment
            )
            
        except Exception as e:
            return EvaluationResult(
                key=evaluator_name,
                score=0.0,
                comment=f"Evaluation error: {str(e)}"
            )
    
    return sync_sampled_evaluator


# No need for custom callback handler - use standard LangChain approach


# =====================================
# NEW ARCHITECTURE: Separate Services
# =====================================

class LLMInstanceManager:
    """Manages LLM instances to avoid recreation"""
    
    def __init__(self):
        self._llm_cache = {}
    
    def get_llm(self, provider: str, model: str, inference_config: Dict[str, Any]) -> ChatGroq:
        """Get or create LLM instance"""
        cache_key = f"{provider}_{model}_{hash(str(sorted(inference_config.items())))}"
        
        if cache_key not in self._llm_cache:
            if provider.upper() != "GROQ":
                raise ValueError(f"Unsupported provider: {provider}")
            
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable is required")
            
            self._llm_cache[cache_key] = ChatGroq(
                groq_api_key=api_key,
                model_name=model,
                temperature=inference_config.get("temperature", 0.0),
                max_tokens=inference_config.get("max_tokens", 150),
                model_kwargs={"seed": inference_config.get("seed", 42)}
            )
        
        return self._llm_cache[cache_key]


class OnDemandEvaluationService:
    """Handles on-demand LLM evaluations"""
    
    def __init__(self, config_path: str, llm_manager: LLMInstanceManager):
        self.config = load_yaml(config_path)
        self.llm_manager = llm_manager
        self.evaluators = {}
        self._initialize_evaluators()
    
    def _initialize_evaluators(self):
        """Initialize LangChain evaluators"""
        evaluator_configs = self.config.get("response_evaluators", [])
        
        for config in evaluator_configs:
            name = config.get("name")
            evaluator_type = config.get("type")
            
            # Get cached LLM instance
            llm = self.llm_manager.get_llm(
                provider=config.get("provider", "GROQ"),
                model=config.get("model", "llama3-8b-8192"),
                inference_config=config.get("inference", {})
            )
            
            # Create custom prompt template if provided
            custom_prompt = None
            if "prompt_template" in config:
                custom_prompt = PromptTemplate.from_template(config["prompt_template"])
            
            # Create evaluator based on type
            custom_criteria = {config.get("name", "custom"): config["criteria"]}
            
            if evaluator_type == "criteria":
                evaluator = CriteriaEvalChain.from_llm(
                    llm=llm,
                    criteria=custom_criteria,
                    prompt=custom_prompt
                )
            elif evaluator_type == "score_string":
                evaluator = ScoreStringEvalChain.from_llm(
                    llm=llm,
                    criteria=custom_criteria,
                    prompt=custom_prompt
                )
            else:
                raise ValueError(f"Unsupported evaluator type: {evaluator_type}")
            
            self.evaluators[name] = {
                "evaluator": evaluator,
                "type": evaluator_type,
                "config": config
            }
    
    async def evaluate(self, prompt: str, response: str, evaluators: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate response using specified evaluators"""
        results: Dict[str, Any] = {}
        
        # Determine which evaluators to run
        if not evaluators:
            evaluators_to_run = self.evaluators.items()
        else:
            evaluators_to_run = [(name, info) for name, info in self.evaluators.items() 
                                if name in evaluators]
        
        for name, evaluator_info in evaluators_to_run:
            evaluator = evaluator_info["evaluator"]
            evaluator_type = evaluator_info["type"]
            
            try:
                # Run LangChain evaluator (NOW PROPERLY ASYNC!)
                result = await evaluator.aevaluate_strings(
                    prediction=response,
                    input=prompt
                )
                
                # Parse to consistent format
                parsed_result = self._parse_langchain_output(result, evaluator_type, name)
                results[name] = parsed_result
                
            except OutputParserException as e:
                results[name] = {
                    "error": "output_parser_error",
                    "details": str(e),
                    "evaluator": name
                }
            except Exception as e:
                results[name] = {
                    "error": "evaluation_failed",
                    "details": str(e),
                    "evaluator": name
                }
        
        return results
    
    def _parse_langchain_output(self, result: Dict[str, Any], evaluator_type: str, evaluator_name: str) -> Dict[str, Any]:
        """Parse LangChain evaluator output to consistent format"""
        try:
            if evaluator_type == "criteria":
                return {
                    "decision": result.get("value", "N"),
                    "score": result.get("score", 0),
                    "evaluation": result.get("reasoning", ""),
                    "evaluator": evaluator_name
                }
            elif evaluator_type == "score_string":
                return {
                    "decision": result.get("score", 1),
                    "score": result.get("score", 1),
                    "evaluation": result.get("reasoning", ""),
                    "evaluator": evaluator_name
                }
            else:
                return {
                    "error": f"unknown_evaluator_type_{evaluator_type}",
                    "raw": str(result),
                    "evaluator": evaluator_name
                }
        except Exception as e:
            return {
                "error": "parsing_failed",
                "details": str(e),
                "raw": str(result),
                "evaluator": evaluator_name
            }
    
    def get_evaluator_names(self) -> List[str]:
        """Get list of configured evaluator names"""
        return list(self.evaluators.keys())
    
    def get_langsmith_evaluators(self):
        """Get evaluators compatible with LangSmith dataset evaluation"""
        return [info["evaluator"] for info in self.evaluators.values()]


class RealtimeEvaluationManager:
    """Simple real-time evaluation with sync evaluators only"""
    
    def __init__(self, realtime_config_path: str, on_demand_service: OnDemandEvaluationService):
        self.realtime_config = {}
        if os.path.exists(realtime_config_path):
            self.realtime_config = load_yaml(realtime_config_path)
        
        self.on_demand_service = on_demand_service
        self.all_evaluators = []  # All sync evaluators
        self._callback_cache = None
        self._initialize_evaluators()
    
    def _initialize_evaluators(self):
        """Initialize ALL evaluators as sync (simple and works!)"""
        realtime_feedback = self.realtime_config.get("realtime_feedback", {})
        
        # Add lightweight evaluators
        lightweight_names = realtime_feedback.get("lightweight", [])
        for name in lightweight_names:
            if name == "response_length":
                self.all_evaluators.append(response_length_evaluator)
            elif name == "language_detection":
                self.all_evaluators.append(language_detection_evaluator)
            elif name == "content_safety":
                self.all_evaluators.append(content_safety_evaluator)
        
        # Add LLM-based evaluators as SYNC with sampling
        llm_sampled = realtime_feedback.get("llm_sampled", [])
        for sampled_config in llm_sampled:
            name = sampled_config.get("name")
            sampling_rate = sampled_config.get("sampling_rate", 0.1)
            
            # Get the evaluator from on-demand service and make it sync
            if name in self.on_demand_service.evaluators:
                evaluator_info = self.on_demand_service.evaluators[name]
                sync_evaluator = create_sync_sampled_evaluator(
                    evaluator_name=name,
                    evaluator=evaluator_info["evaluator"],
                    sampling_rate=sampling_rate
                )
                self.all_evaluators.append(sync_evaluator)
    
    def get_callbacks(self, project_name: str) -> List:
        """Get simple, working callback handlers"""
        if not self.all_evaluators:
            return []
        
        # Cache callbacks to avoid recreation
        if self._callback_cache is None:
            # Set LangSmith project
            os.environ["LANGSMITH_PROJECT"] = project_name
            
            # Use standard LangChain callback handler (simple and works!)
            from langchain_core.tracers.evaluation import EvaluatorCallbackHandler
            self._callback_cache = [EvaluatorCallbackHandler(evaluators=self.all_evaluators)]
        
        return self._callback_cache


class FeedbackService:
    """Handles human feedback management"""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.langsmith_client = None
        if os.getenv("LANGSMITH_TRACING") == "true":
            self.langsmith_client = Client()
    
    async def record_feedback(self, run_id: str, feedback_type: str, value: Any, comment: Optional[str] = None) -> Dict[str, str]:
        """Record human feedback to LangSmith"""
        if not self.langsmith_client:
            return {"status": "error", "message": "LangSmith not configured"}
        
        try:
            # Process feedback value based on type
            if feedback_type == "thumbs":
                score = 1.0 if value == "up" else 0.0
                processed_value = value
            elif feedback_type == "score":
                score = float(value) / 5.0  # Normalize 1-5 to 0-1
                processed_value = value
            else:
                score = None
                processed_value = value
            
            # Send to LangSmith
            self.langsmith_client.create_feedback(
                run_id=run_id,
                key=feedback_type,
                score=score,
                value=processed_value,
                comment=comment
            )
            
            return {"status": "recorded", "run_id": run_id}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}


class EvaluationService:
    """
    Clean coordinator that delegates to specialized services:
    - OnDemandEvaluationService: Handles LLM evaluations
    - RealtimeEvaluationManager: Orchestrates real-time feedback
    - FeedbackService: Manages human feedback
    """
    
    def __init__(self, project_name: str, config_path: str, realtime_config_path: Optional[str] = None):
        self.project_name = project_name
        realtime_config_path = realtime_config_path or f"configs/evaluators/{project_name}_realtime.yaml"
        
        # Initialize shared LLM manager
        self.llm_manager = LLMInstanceManager()
        
        # Initialize specialized services
        self.on_demand_service = OnDemandEvaluationService(config_path, self.llm_manager)
        self.realtime_manager = RealtimeEvaluationManager(realtime_config_path, self.on_demand_service)
        self.feedback_service = FeedbackService(project_name)
    
    # Delegate to specialized services
    
    def get_realtime_callbacks(self) -> List:
        """Get callback handlers for real-time automated feedback"""
        return self.realtime_manager.get_callbacks(self.project_name)
    
    async def evaluate_on_demand(self, prompt: str, response: str, evaluators: Optional[List[str]] = None) -> Dict[str, Any]:
        """On-demand evaluation using full LLM evaluation suite"""
        # Set LangSmith project before evaluation
        os.environ["LANGSMITH_PROJECT"] = self.project_name
        return await self.on_demand_service.evaluate(prompt, response, evaluators)
    
    async def record_human_feedback(self, run_id: str, feedback_type: str, value: Any, comment: Optional[str] = None) -> Dict[str, str]:
        """Record human feedback and send to LangSmith"""
        return await self.feedback_service.record_feedback(run_id, feedback_type, value, comment)
    
    def get_evaluator_names(self) -> List[str]:
        """Get list of configured evaluator names"""
        return self.on_demand_service.get_evaluator_names()
    
    def get_langsmith_evaluators(self):
        """Get evaluators compatible with LangSmith dataset evaluation"""
        return self.on_demand_service.get_langsmith_evaluators()