import os
from typing import Any, Dict

from langchain_groq import ChatGroq
from langchain.evaluation.criteria import CriteriaEvalChain
from langchain.evaluation.scoring import ScoreStringEvalChain
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException

from ..utils.config_loader import load_yaml


class ResponseEvaluatorService:
    """
    LangChain-native service to evaluate model responses using standard evaluators.
    
    Features:
    - Uses native LangChain evaluators (CriteriaEvaluator, ScoreStringEvaluator)
    - Custom prompt templates in Portuguese 
    - Independent service with no external dependencies
    - Compatible with LangSmith for dataset evaluation
    """
    
    def __init__(self, config_path: str):
        self.config = load_yaml(config_path)
        self.evaluator_configs = self.config.get("response_evaluators", [])
        
        # Validate configuration
        if not self.evaluator_configs:
            raise ValueError("No evaluators configured in response_evaluators")
        
        # Initialize evaluators
        self.evaluators = {}
        self._initialize_evaluators()
    
    def _create_llm(self, evaluator_config: Dict[str, Any]) -> ChatGroq:
        """Create LLM instance for a specific evaluator"""
        provider = evaluator_config.get("provider", "GROQ").upper()
        
        if provider != "GROQ":
            raise ValueError(f"Unsupported provider: {provider}")
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        model = evaluator_config.get("model", "llama3-8b-8192")
        inference_config = evaluator_config.get("inference", {})
        
        return ChatGroq(
            groq_api_key=api_key,
            model_name=model,
            temperature=inference_config.get("temperature", 0.0),
            max_tokens=inference_config.get("max_tokens", 150),
            model_kwargs={"seed": inference_config.get("seed", 42)}
        )
    
    def _create_prompt_template(self, prompt_template_str: str) -> PromptTemplate:
        """Create LangChain PromptTemplate from string"""
        return PromptTemplate.from_template(prompt_template_str)
    
    def _initialize_evaluators(self):
        """Initialize LangChain evaluators based on configuration"""
        for config in self.evaluator_configs:
            name = config.get("name")
            evaluator_type = config.get("type")
            
            # Create LLM for this evaluator
            llm = self._create_llm(config)
            
            # Create custom prompt template if provided
            custom_prompt = None
            if "prompt_template" in config:
                custom_prompt = self._create_prompt_template(config["prompt_template"])
            
            # Create evaluator based on type
            # Use custom criteria format: {name: description}
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
                raise ValueError(f"Unsupported evaluator type: {evaluator_type}. Supported types: ['criteria', 'score_string']")
            
            self.evaluators[name] = {
                "evaluator": evaluator,
                "type": evaluator_type,
                "config": config
            }
    
    def _parse_langchain_output(self, result: Dict[str, Any], evaluator_type: str, evaluator_name: str) -> Dict[str, Any]:
        """Parse LangChain evaluator output to consistent format"""
        try:
            if evaluator_type == "criteria":
                # CriteriaEvaluator returns: {"score": 0/1, "value": "Y"/"N", "reasoning": "..."}
                return {
                    "decision": result.get("value", "N"),  # Y/N format
                    "score": result.get("score", 0),      # 0/1 binary
                    "evaluation": result.get("reasoning", ""),
                    "evaluator": evaluator_name
                }
            elif evaluator_type == "score_string":
                # ScoreStringEvaluator returns: {"score": 1-10, "reasoning": "..."}
                return {
                    "decision": result.get("score", 1),   # 1-10 score
                    "score": result.get("score", 1),     # Same as decision for consistency
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
    
    async def evaluate_response(self, prompt: str, response: str, evaluators: list[str] = None) -> Dict[str, Any]:
        """
        Evaluate the given model response against the user prompt for specified evaluators.
        
        Args:
            prompt: User prompt/input
            response: Model response to evaluate
            evaluators: List of evaluator names to run. If None or empty, runs all evaluators.
        
        Returns:
            Mapping from evaluator name to evaluation result.
        """
        results: Dict[str, Any] = {}
        
        # Determine which evaluators to run
        if not evaluators:
            # Run all evaluators (None or empty list)
            evaluators_to_run = self.evaluators.items()
        else:
            # Run only specified evaluators
            evaluators_to_run = [(name, info) for name, info in self.evaluators.items() 
                                if name in evaluators]
        
        for name, evaluator_info in evaluators_to_run:
            evaluator = evaluator_info["evaluator"]
            evaluator_type = evaluator_info["type"]
            
            try:
                # Run LangChain evaluator
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
    
    async def evaluate_single(self, evaluator_name: str, prompt: str, response: str) -> Dict[str, Any]:
        """
        Evaluate using a single evaluator by name.
        Useful for testing or selective evaluation.
        """
        if evaluator_name not in self.evaluators:
            return {
                "error": "evaluator_not_found",
                "evaluator": evaluator_name
            }
        
        evaluator_info = self.evaluators[evaluator_name]
        evaluator = evaluator_info["evaluator"]
        evaluator_type = evaluator_info["type"]
        
        try:
            result = await evaluator.aevaluate_strings(
                prediction=response,
                input=prompt
            )
            
            return self._parse_langchain_output(result, evaluator_type, evaluator_name)
            
        except Exception as e:
            return {
                "error": "evaluation_failed",
                "details": str(e),
                "evaluator": evaluator_name
            }
    
    def get_evaluator_names(self) -> list[str]:
        """Get list of configured evaluator names"""
        return list(self.evaluators.keys())
    
    def get_evaluator_info(self, evaluator_name: str) -> Dict[str, Any] | None:
        """Get information about a specific evaluator"""
        if evaluator_name in self.evaluators:
            info = self.evaluators[evaluator_name].copy()
            # Don't expose the actual evaluator object
            info.pop("evaluator", None)
            return info
        return None
    
    def get_langsmith_evaluators(self):
        """
        Get evaluators in format compatible with LangSmith dataset evaluation.
        Returns list of LangChain evaluator objects.
        """
        return [info["evaluator"] for info in self.evaluators.values()]