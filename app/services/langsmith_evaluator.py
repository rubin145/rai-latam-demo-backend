import os
import asyncio
from typing import Dict, Any, Optional
from langsmith import Client
from langchain_groq import ChatGroq
from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example

from .response_evaluator import ResponseEvaluatorService
from ..utils.config_loader import load_yaml


class LangSmithEvaluatorService:
    """
    Service to integrate LLM-as-a-judge evaluators with LangSmith traces.
    
    Features:
    - Uses existing GROQ-based evaluators from ResponseEvaluatorService
    - Automatically evaluates chat responses and adds feedback to traces
    - Shows evaluation results in LangSmith tracing dashboard
    - Uses the same GROQ API key as the main chat model
    """
    
    def __init__(self, evaluator_config_path: str = "configs/evaluators/llm_evaluators.yaml"):
        self.langsmith_client = Client()
        self.response_evaluator = ResponseEvaluatorService(evaluator_config_path)
        self.config = load_yaml(evaluator_config_path)
        
        # Initialize evaluator names for easy access
        self.evaluator_names = self.response_evaluator.get_evaluator_names()
        print(f"🔍 LangSmith LLM-as-a-judge evaluators initialized: {self.evaluator_names}")
    
    async def evaluate_and_add_feedback(
        self, 
        run_id: str, 
        prompt: str, 
        response: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a response using all configured evaluators and add feedback to LangSmith trace.
        
        Args:
            run_id: LangSmith run ID for the trace
            prompt: User's original prompt/query
            response: Model's response to evaluate
            session_id: Session identifier for context
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            # Run all evaluations
            evaluation_results = await self.response_evaluator.evaluate_response(prompt, response)
            
            # Add feedback to LangSmith trace for each evaluator
            feedback_tasks = []
            for evaluator_name, result in evaluation_results.items():
                if not result.get("error"):
                    feedback_tasks.append(
                        self._add_feedback_to_trace(run_id, evaluator_name, result, session_id)
                    )
            
            # Execute all feedback additions in parallel
            if feedback_tasks:
                await asyncio.gather(*feedback_tasks, return_exceptions=True)
            
            print(f"✅ Added LLM-as-a-judge feedback to trace {str(run_id)[:8]}... for evaluators: {list(evaluation_results.keys())}")
            return evaluation_results
            
        except Exception as e:
            print(f"❌ Error evaluating response for trace {str(run_id)}: {e}")
            return {"error": str(e)}
    
    async def _add_feedback_to_trace(
        self, 
        run_id: str, 
        evaluator_name: str, 
        evaluation_result: Dict[str, Any],
        session_id: Optional[str] = None
    ):
        """Add feedback to a specific LangSmith trace."""
        try:
            # Prepare feedback data based on evaluator type
            if evaluation_result.get("decision") in ["YES", "NO"]:
                # Criteria evaluator (binary YES/NO)
                score = 1.0 if evaluation_result["decision"] == "YES" else 0.0
                value = evaluation_result["decision"]
                comment = evaluation_result.get("evaluation", "")
                
            elif isinstance(evaluation_result.get("decision"), (int, float)):
                # Score string evaluator (1-3 scale for hallucination)
                score = float(evaluation_result["decision"]) / 3.0  # Normalize to 0-1 (3=worst, 1=best)
                value = evaluation_result["decision"]
                comment = evaluation_result.get("evaluation", "")
                
            else:
                # Fallback
                score = evaluation_result.get("score", 0.0)
                value = evaluation_result.get("decision", "unknown")
                comment = evaluation_result.get("evaluation", "")
            
            # Create feedback
            feedback = self.langsmith_client.create_feedback(
                run_id=run_id,
                key=f"llm_judge_{evaluator_name}",
                score=score,
                value=value,
                comment=comment,
                metadata={
                    "evaluator_type": "llm_as_judge",
                    "evaluator_name": evaluator_name,
                    "session_id": session_id,
                    "model": "groq_llama3"
                }
            )
            
            return feedback
            
        except Exception as e:
            print(f"❌ Error adding feedback for {evaluator_name} to trace {str(run_id)}: {e}")
            return None
    
    def create_langsmith_evaluators(self):
        """
        Create LangSmith-compatible evaluator functions for use with langsmith.evaluate().
        Returns a list of evaluator functions that can be used in LangSmith datasets.
        """
        evaluator_functions = []
        
        for evaluator_name in self.evaluator_names:
            async def evaluator_func(run: Run, example: Example, evaluator_name=evaluator_name):
                """LangSmith evaluator function"""
                try:
                    # Extract inputs and outputs from the run
                    inputs = run.inputs or {}
                    outputs = run.outputs or {}
                    
                    prompt = inputs.get("query", inputs.get("input", ""))
                    response = outputs.get("response", outputs.get("output", ""))
                    
                    if not prompt or not response:
                        return {"key": f"llm_judge_{evaluator_name}", "score": None, "comment": "Missing input or output"}
                    
                    # Run evaluation
                    result = await self.response_evaluator.evaluate_single(evaluator_name, prompt, response)
                    
                    # Format result for LangSmith
                    if result.get("error"):
                        return {"key": f"llm_judge_{evaluator_name}", "score": None, "comment": f"Error: {result['error']}"}
                    
                    if result.get("decision") in ["YES", "NO"]:
                        score = 1.0 if result["decision"] == "YES" else 0.0
                    elif isinstance(result.get("decision"), (int, float)):
                        score = float(result["decision"]) / 3.0  # Normalize 1-3 scale
                    else:
                        score = result.get("score", 0.0)
                    
                    return {
                        "key": f"llm_judge_{evaluator_name}",
                        "score": score,
                        "comment": result.get("evaluation", "")
                    }
                    
                except Exception as e:
                    return {"key": f"llm_judge_{evaluator_name}", "score": None, "comment": f"Evaluation error: {str(e)}"}
            
            evaluator_functions.append(evaluator_func)
        
        return evaluator_functions
    
    def get_available_evaluators(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available evaluators."""
        return {
            name: self.response_evaluator.get_evaluator_info(name) 
            for name in self.evaluator_names
        }
    
    async def evaluate_single_trace(
        self, 
        run_id: str, 
        evaluator_names: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single trace by run_id using specified evaluators.
        Useful for retroactive evaluation of existing traces.
        """
        try:
            # Get the run from LangSmith
            run = self.langsmith_client.read_run(run_id)
            
            # Extract prompt and response
            inputs = run.inputs or {}
            outputs = run.outputs or {}
            
            prompt = inputs.get("query", inputs.get("input", ""))
            response = outputs.get("response", outputs.get("output", ""))
            
            if not prompt or not response:
                return {"error": "Could not extract prompt and response from trace"}
            
            # Run evaluations
            if evaluator_names:
                results = {}
                for evaluator_name in evaluator_names:
                    if evaluator_name in self.evaluator_names:
                        result = await self.response_evaluator.evaluate_single(evaluator_name, prompt, response)
                        results[evaluator_name] = result
                        
                        # Add feedback if successful
                        if not result.get("error"):
                            await self._add_feedback_to_trace(run_id, evaluator_name, result)
                return results
            else:
                # Run all evaluators
                return await self.evaluate_and_add_feedback(run_id, prompt, response)
                
        except Exception as e:
            print(f"❌ Error evaluating trace {str(run_id)}: {e}")
            return {"error": str(e)} 