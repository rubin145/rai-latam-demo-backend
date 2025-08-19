import os
import asyncio
from typing import Dict, Any, Optional

from langsmith import Client
from langsmith.schemas import Run, Example
from ..utils.evaluate import normalize_score
from .evaluators import LLMEvaluator, LightEvaluator

class LangSmithClient:
    """
    Service to integrate LLM-as-a-judge evaluators and Light evaluators with LangSmith traces.
    
    Features:
    - Uses existing GROQ-based evaluators from ResponseEvaluatorService
    - Automatically evaluates chat responses and adds feedback to traces
    - Shows evaluation results in LangSmith tracing dashboard
    - Uses the same GROQ API key as the main chat model
    """
    
    def __init__(self, evaluator_config_path: str = "configs/evaluators/llm_evaluators.yaml"):
        self.langsmith_client = Client()
        self.evaluator_manager = LLMEvaluator(evaluator_config_path)
        self.light_evaluator = LightEvaluator()
        
        # Initialize evaluator names for easy access
        self.evaluator_names = self.evaluator_manager.get_evaluator_names()
        print(f"🔍 LangSmith LLM-as-a-judge evaluators initialized: {self.evaluator_names}")
        
    def _format_llm_feedback(self, evaluator_name: str, evaluation_result: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """Format LLM feedback data consistently for LangSmith."""
        score = evaluation_result.get("score", 0.0)
        value = evaluation_result.get("decision", "unknown")
        comment = evaluation_result.get("evaluation", "")

        if evaluation_result.get("decision") in ["YES", "NO"]:
            score = 1.0 if evaluation_result["decision"] == "YES" else 0.0
            value = evaluation_result["decision"]
        elif isinstance(evaluation_result.get("decision"), (int, float)):
            score = normalize_score(evaluation_result["decision"])
            value = evaluation_result["decision"]
        elif "score" in evaluation_result and "comment" in evaluation_result:
            score = evaluation_result["score"]
            value = evaluation_result["score"]
            comment = evaluation_result["comment"]

        return {
            "key": f"llm_judge_{evaluator_name}",
            "score": score,
            "value": value,
            "comment": comment,
            "metadata": {
                "evaluator_type": "llm_as_judge",
                "evaluator_name": evaluator_name,
                "session_id": session_id,
                "model": "groq_llama3"
            }
        }

    def _format_light_feedback(self, evaluator_name: str, evaluation_result: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """Formats a single result from LightEvaluator for submission to LangSmith."""
        if "error" in evaluation_result:
            return {
                "key": f"lightweight_{evaluator_name}",
                "score": 0,
                "comment": f"Error during evaluation: {evaluation_result['error']}",
                "metadata": {
                    "evaluator_type": "heuristic",
                    "evaluator_name": evaluator_name,
                    "session_id": session_id
                }
            }
        else:
            return {
                "key": f"lightweight_{evaluator_name}",
                "score": evaluation_result.get("score"),
                "value": evaluation_result.get("value"),
                "comment": evaluation_result.get("comment"),
                "metadata": {
                    "evaluator_type": "heuristic",
                    "evaluator_name": evaluator_name,
                    "session_id": session_id
                }
            }
    
    async def record_human_feedback(
        self, 
        run_id: str, 
        feedback_type: str, 
        value: Any, 
        comment: Optional[str] = None
    ) -> Dict[str, str]:
        """Record human feedback to LangSmith trace."""
        try:
            # Process feedback value based on type
            if feedback_type == "thumbs":
                score = 1.0 if value == "up" else 0.0
            elif feedback_type == "rating":
                score = normalize_score(value, scale=5)  # Normalize 1-5 to 0-1
            else:
                score = None

            # Format feedback data
            feedback_data = {
                "key": f"human_{feedback_type}",
                "score": score,
                "value": value,
                "comment": comment,
                "metadata": {
                    "feedback_source": "human",
                    "feedback_type": feedback_type
                }
            }

            # Send to LangSmith
            self.langsmith_client.create_feedback(
                run_id=run_id,
                key=feedback_data["key"],
                score=feedback_data["score"],
                value=feedback_data["value"],
                comment=feedback_data["comment"],
                metadata=feedback_data["metadata"]
            )

            return {"status": "recorded", "run_id": run_id}

        except Exception as e:
            return {"status": "error", "message": str(e)}
    
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
            # Run lightweight evaluations immediately (no API calls)
            lightweight_results = self.light_evaluator.run_evaluations(response)

            # Add lightweight feedback to trace
            for eval_name, result in lightweight_results.items():
                if not result.get("error"):
                    await self._add_feedback_to_trace(run_id, f"lightweight_{eval_name}", result, session_id)

            # Run LLM evaluations
            llm_evaluation_results = await self.evaluator_manager.evaluate_response(prompt, response)

            # Add LLM feedback to LangSmith trace
            feedback_tasks = [
                self._add_feedback_to_trace(run_id, evaluator_name, result, session_id)
                for evaluator_name, result in llm_evaluation_results.items()
                if not result.get("error")
            ]

            # Combine all results for return
            evaluation_results = {**llm_evaluation_results, **lightweight_results}

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
            # Format feedback based on evaluator type
            if "lightweight" in evaluator_name:
                feedback_data = self._format_light_feedback(evaluator_name.replace("lightweight_", ""), evaluation_result, session_id)
            else:
                feedback_data = self._format_llm_feedback(evaluator_name, evaluation_result, session_id)

            # Create feedback
            feedback = self.langsmith_client.create_feedback(
                run_id=run_id,
                key=feedback_data["key"],
                score=feedback_data["score"],
                value=feedback_data["value"],
                comment=feedback_data["comment"],
                metadata=feedback_data["metadata"]
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
                    result = await self.evaluator_manager.evaluate_single(evaluator_name, prompt, response)
                    
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
            name: self.evaluator_manager.get_evaluator_info(name) 
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
                        result = await self.evaluator_manager.evaluate_single(evaluator_name, prompt, response)
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

    async def evaluate_trace_readonly(
        self, 
        run_id: str, 
        evaluator_names: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single trace by run_id WITHOUT adding feedback to LangSmith.
        Useful for analysis without modifying the trace.
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
            
            # Run evaluations WITHOUT adding feedback
            if evaluator_names:
                results = {}
                for evaluator_name in evaluator_names:
                    if evaluator_name in self.evaluator_names:
                        result = await self.evaluator_manager.evaluate_single(evaluator_name, prompt, response)
                        results[evaluator_name] = result
                return results
            else:
                # Run all evaluators WITHOUT adding feedback
                return await self.evaluator_manager.evaluate_response(prompt, response)
                
        except Exception as e:
            print(f"❌ Error evaluating trace {str(run_id)}: {e}")
            return {"error": str(e)}

    async def evaluate_dataset(
        self, 
        dataset_id: str, 
        evaluator_names: Optional[list] = None,
        add_feedback: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate all examples in a LangSmith dataset.
        
        Args:
            dataset_id: LangSmith dataset ID
            evaluator_names: List of evaluator names to use (default: all)
            add_feedback: Whether to add feedback to traces (default: True)
        """
        try:
            # Get dataset from LangSmith
            dataset = self.langsmith_client.read_dataset(dataset_name=dataset_id)
            examples = list(self.langsmith_client.list_examples(dataset_id=dataset.id))
            
            if not examples:
                return {"error": f"No examples found in dataset {dataset_id}"}
            
            results = {
                "dataset_id": dataset_id,
                "dataset_name": dataset.name,
                "total_examples": len(examples),
                "evaluations": {}
            }
            
            # Evaluate each example
            for example in examples:
                example_id = str(example.id)
                
                # Extract inputs/outputs
                inputs = example.inputs or {}
                outputs = example.outputs or {}
                
                prompt = inputs.get("query", inputs.get("input", ""))
                response = outputs.get("response", outputs.get("output", ""))
                
                if prompt and response:
                    # Run evaluations
                    if evaluator_names:
                        eval_results = {}
                        for evaluator_name in evaluator_names:
                            if evaluator_name in self.evaluator_names:
                                result = await self.evaluator_manager.evaluate_single(evaluator_name, prompt, response)
                                eval_results[evaluator_name] = result
                    else:
                        eval_results = await self.evaluator_manager.evaluate_response(prompt, response)
                    
                    results["evaluations"][example_id] = {
                        "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                        "response": response[:100] + "..." if len(response) > 100 else response,
                        "evaluations": eval_results
                    }
                    
                    # Optionally add feedback if there's a run_id
                    if add_feedback and hasattr(example, 'run_id') and example.run_id:
                        for evaluator_name, result in eval_results.items():
                            if not result.get("error"):
                                await self._add_feedback_to_trace(example.run_id, evaluator_name, result)
            
            return results
                
        except Exception as e:
            print(f"❌ Error evaluating dataset {dataset_id}: {e}")
            return {"error": str(e)}





