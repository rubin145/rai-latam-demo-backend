import os
from langchain_groq import ChatGroq
from typing import Dict, Any

class LLMManager:
    """
    Manages caching of LLM instances to avoid recreation.

    Features:
    - Validates provider and API key.
    - Retrieves and caches LLM instances based on configuration.
    """
    def __init__(self):
        self._llm_cache = {}

    def get_llm(self, provider: str, model: str, inference_config: Dict[str, Any], **kwargs) -> ChatGroq:
        """
        Retrieve or create a cached LLM instance.

        Args:
            provider: LLM provider name (e.g., "GROQ").
            model: Model name (e.g., "llama3-8b-8192").
            inference_config: Configuration for inference (e.g., temperature, max_tokens).

        Returns:
            Cached ChatGroq instance.

        Raises:
            ValueError: If provider is unsupported or API key is missing.
        """
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