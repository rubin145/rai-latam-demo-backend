import json
from typing import Any, Dict

from groq import Groq

from ..utils.config_loader import load_yaml


class ResponseEvaluatorService:
    """
    Service to evaluate a model response across multiple dimensions defined in a Groq config.
    """
    def __init__(self, api_key: str, config_path: str):
        self.client = Groq(api_key=api_key)
        cfg = load_yaml(config_path)
        self.evaluators = cfg.get("response_evaluators", [])

    async def evaluate_response(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Evaluate the given model response against the user prompt for each configured dimension.
        Returns a mapping from evaluator name to its parsed JSON output (or raw text on failure).
        """
        results: Dict[str, Any] = {}
        for evaluator in self.evaluators:
            name = evaluator.get("name")
            system_prompt = evaluator.get("system_prompt", "")
            model = evaluator.get("model")
            inference = evaluator.get("inference", {})

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append(
                {
                    "role": "user",
                    "content": f"Prompt:\n{prompt.strip()}\n\nResponse:\n{response.strip()}"
                }
            )

            resp = self.client.chat.completions.create(
                model=model, messages=messages, **inference
            )
            content = ""
            if hasattr(resp, "choices") and resp.choices:
                content = resp.choices[0].message.content

            try:
                parsed = json.loads(content)
            except Exception:
                parsed = {"error": "failed_to_parse", "raw": content}

            results[name] = parsed

        return results