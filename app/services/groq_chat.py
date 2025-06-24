import uuid
from typing import Tuple

from groq import Groq

from ..utils.config_loader import load_yaml

class GroqChatService:
    """Chat service implementation using the Groq SDK."""
    def __init__(self, api_key: str, config_path: str):
        self.client = Groq(api_key=api_key)
        cfg = load_yaml(config_path)
        self.system_prompt = cfg.get("system_prompt", "")
        self.model = cfg.get("model", "")
        self.inference = cfg.get("inference", {})
        # Maximum number of messages to retain in conversation context (sliding window)
        self.max_history = cfg.get("max_history", 20)
        # Local history buffer to preserve conversation context per session
        self.chat_history: dict[str, list[dict]] = {}

    async def handle_chat(self, query: str, session_id: str = None) -> Tuple[str, str]:
        if not session_id:
            session_id = str(uuid.uuid4())

        history = self.chat_history.setdefault(session_id, [])
        if self.system_prompt and not any(m.get("role") == "system" for m in history):
            history.append({"role": "system", "content": self.system_prompt})
        history.append({"role": "user", "content": query.strip()})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=history,
            **self.inference,
        )
        content = ""
        if hasattr(response, "choices") and response.choices:
            content = response.choices[0].message.content
        history.append({"role": "assistant", "content": content})

        # Trim history to the most recent messages (preserving system prompt)
        if len(history) > self.max_history:
            new_history = []
            if self.system_prompt and history and history[0].get("role") == "system":
                new_history.append(history[0])
            tail_size = self.max_history - len(new_history)
            if tail_size > 0:
                new_history.extend(history[-tail_size:])
            history[:] = new_history

        return content, session_id