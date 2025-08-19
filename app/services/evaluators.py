from typing import Dict, Any
from langchain.evaluation.criteria import CriteriaEvalChain
from langchain.evaluation.scoring import ScoreStringEvalChain
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException

from ..utils.config_loader import load_yaml
from .llm_manager import LLMManager

class LLMEvaluator:
    """
    LangChain-native service to evaluate model responses using standard evaluators.

    Features:
    - Uses native LangChain evaluators (CriteriaEvaluator, ScoreStringEvaluator)
    - Custom prompt templates
    - Compatible with LangSmith for dataset evaluation and for real-time feedback
    """
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to evaluator configuration file.

        Raises:
            ValueError: If no evaluators are configured.
        """
        self.config = load_yaml(config_path)
        self.evaluator_configs = self.config.get("response_evaluators", [])
        self.llm_manager = LLMManager()

        if not self.evaluator_configs:
            raise ValueError("No evaluators configured in response_evaluators")

        self.evaluators = {}
        self._initialize_evaluators()

    def _initialize_evaluators(self):
        """
        Initialize evaluators based on configuration.

        Raises:
            ValueError: If evaluator type is unsupported.
        """
        for config in self.evaluator_configs:
            name = config.get("name")
            evaluator_type = config.get("type")
            provider = config.get("provider", "GROQ")
            model = config.get("model", "llama3-8b-8192")
            inference_config = config.get("inference", {})

            llm = self.llm_manager.get_llm(provider, model, inference_config)

            # Create custom prompt template if provided
            custom_prompt = None
            if "prompt_template" in config:
                custom_prompt = PromptTemplate.from_template(config["prompt_template"])

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


class LightEvaluator:
    """
    Class to handle lightweight evaluations.
    """
    def __init__(self):
        self.evaluators = {
            'response_length': self._evaluate_response_length,
            'language_detection': self._evaluate_language,
            'content_safety': self._evaluate_content_safety
        }

    def _evaluate_response_length(self, response: str) -> Dict[str, Any]:
        """Evaluate response length (10-2000 characters optimal)"""
        length = len(response)
        score = 1.0 if 10 <= length <= 2000 else 0.3
        comment = f"Response length: {length} characters"
        if length < 10:
            comment += " (too short)"
        elif length > 2000:
            comment += " (too long)"
        else:
            comment += " (optimal)"

        return {
            "score": score,
            "comment": comment,
            "evaluator": "response_length"
        }

    def _evaluate_language(self, response: str) -> Dict[str, Any]:
        """Basic language detection using character patterns"""
        expected_language = "pt"

        # Simple Portuguese indicators
        pt_indicators = ['ã', 'ç', 'ção', 'ões', 'ão', 'em', 'de', 'do', 'da', 'para', 'com']

        if expected_language == "pt":
            pt_count = sum(1 for indicator in pt_indicators if indicator in response.lower())
            score = min(pt_count / 3, 1.0)  # Normalize to 0-1
        else:
            score = 1.0  # Default pass for other languages

        return {
            "score": score,
            "comment": f"Language detection score: {score:.2f} for {expected_language}",
            "evaluator": "language_detection"
        }

    def _evaluate_content_safety(self, response: str) -> Dict[str, Any]:
        """Basic content safety using regex patterns"""
        import re

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

        return {
            "score": score,
            "comment": comment,
            "evaluator": "content_safety"
        }

    def run_evaluations(self, response: str) -> Dict[str, Any]:
        """Run all lightweight evaluations."""
        results = {}
        for name, evaluator_func in self.evaluators.items():
            try:
                results[name] = evaluator_func(response)
            except Exception as e:
                results[name] = {"error": str(e), "evaluator": name}
        return results
