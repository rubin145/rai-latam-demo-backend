import json
from typing import Any, Dict

from langchain.schema import SystemMessage, HumanMessage
from .langchain_chat import LangChainChatService
from ..utils.config_loader import load_yaml


class ResponseEvaluatorService:
    """
    Service to evaluate a model response across multiple dimensions using LangChain.
    """
    def __init__(self, config_path: str):
        cfg = load_yaml(config_path)
        self.evaluators = cfg.get("response_evaluators", [])
        self.config = cfg
        
        # Create LangChain service for evaluation
        self.langchain_service = LangChainChatService(config_path)

    async def evaluate_response(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Evaluate the given model response against the user prompt for each configured dimension.
        Returns a mapping from evaluator name to its parsed JSON output (or raw text on failure).
        """
        results: Dict[str, Any] = {}
        for evaluator in self.evaluators:
            name = evaluator.get("name")
            system_prompt = evaluator.get("system_prompt", "")
            model = evaluator.get("model", self.config.get("model", "llama3-8b-8192"))
            inference = evaluator.get("inference", {})

            # Create specialized LLM for this evaluator
            evaluator_llm = self.langchain_service._create_filter_llm({
                "model": model,
                "inference": inference
            })

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Prompt:\n{prompt.strip()}\n\nResponse:\n{response.strip()}")
            ]

            try:
                resp = await evaluator_llm.ainvoke(messages)
                content = resp.content
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = {"error": "failed_to_parse", "raw": content}
            except Exception as e:
                parsed = {"error": "evaluation_failed", "details": str(e)}

            results[name] = parsed

        return results