import uuid
import os
import json
from typing import Tuple, Dict, Any

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage, SystemMessage

from ..utils.config_loader import load_yaml


class LangChainChatService:
    """Generic chat service implementation using LangChain with multiple providers."""
    
    def __init__(self, config_path: str):
        cfg = load_yaml(config_path)
        
        # Get provider configuration
        provider = cfg.get("provider", "GROQ").upper()
        
        # Initialize LangChain LLM based on provider
        if provider == "GROQ":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable is required for GROQ provider")
            
            self.llm = ChatGroq(
                groq_api_key=api_key,
                model_name=cfg.get("model", "llama3-70b-8192"),
                temperature=cfg.get("inference", {}).get("temperature", 0.7),
                max_tokens=cfg.get("inference", {}).get("max_tokens", 400),
                model_kwargs={"seed": cfg.get("inference", {}).get("seed", 42)}
            )
        
        elif provider == "OPENAI":
            # Future implementation
            # from langchain_openai import ChatOpenAI
            # api_key = os.getenv("OPENAI_API_KEY")
            # if not api_key:
            #     raise ValueError("OPENAI_API_KEY environment variable is required for OPENAI provider")
            # self.llm = ChatOpenAI(api_key=api_key, model=cfg.get("model", "gpt-4"), ...)
            raise NotImplementedError("OpenAI provider not yet implemented")
        
        elif provider == "ANTHROPIC":
            # Future implementation
            # from langchain_anthropic import ChatAnthropic
            # api_key = os.getenv("ANTHROPIC_API_KEY")
            # if not api_key:
            #     raise ValueError("ANTHROPIC_API_KEY environment variable is required for ANTHROPIC provider")
            # self.llm = ChatAnthropic(api_key=api_key, model=cfg.get("model", "claude-3-sonnet"), ...)
            raise NotImplementedError("Anthropic provider not yet implemented")
        
        else:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: GROQ")
        
        # Configuration
        self.provider = provider
        self.system_prompt = cfg.get("system_prompt", "")
        self.max_history = cfg.get("max_history", 20)
        
        # Simple history management per session
        self.chat_history: dict[str, list] = {}
        
        # Load full config for filters
        self.config = cfg
    
    async def apply_input_filters(self, query: str) -> Tuple[str, str, str]:
        """Apply input filters to query. Returns (decision, evaluation, template_response)"""
        input_filters = self.config.get("input_filters", [])
        
        for filter_config in input_filters:
            # Create filter LLM with specific settings
            filter_llm = self._create_filter_llm(filter_config)
            
            # Apply filter
            filter_messages = [
                SystemMessage(content=filter_config["system_prompt"]),
                HumanMessage(content=query.strip())
            ]
            
            try:
                response = await filter_llm.ainvoke(filter_messages)
                filter_result = json.loads(response.content)
                
                if filter_result.get("decision") == "danger":
                    return (
                        "danger",
                        f"{filter_config.get('name', 'filter')}: {filter_result.get('evaluation', '')}",
                        filter_config.get("template_response", "Sorry, I can't help with that.")
                    )
            except (json.JSONDecodeError, Exception) as e:
                # If filter fails, log and continue
                print(f"⚠️ Filter {filter_config.get('name')} failed: {e}")
                continue
        
        return ("safe", "", "")
    
    def _create_filter_llm(self, filter_config: Dict[str, Any]):
        """Create a specialized LLM for filter with specific configuration."""
        if self.provider == "GROQ":
            return ChatGroq(
                groq_api_key=os.getenv("GROQ_API_KEY"),
                model_name=filter_config.get("model", "llama3-70b-8192"),
                temperature=filter_config.get("inference", {}).get("temperature", 0.0),
                max_tokens=filter_config.get("inference", {}).get("max_tokens", 150),
                model_kwargs={"seed": filter_config.get("inference", {}).get("seed", 42)}
            )
        elif self.provider == "OPENAI":
            # Future: Create OpenAI filter LLM
            raise NotImplementedError("OpenAI filter LLM not implemented")
        else:
            raise ValueError(f"Unsupported provider for filters: {self.provider}")
    
    async def handle_chat(self, query: str, session_id: str = None) -> Tuple[str, str]:
        """Handle a chat message and return response with session ID."""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Get or create history for this session
        history = self.chat_history.setdefault(session_id, [])
        
        # Add system prompt if not present and configured
        if self.system_prompt and not any(isinstance(msg, SystemMessage) for msg in history):
            history.insert(0, SystemMessage(content=self.system_prompt))
        
        # Add user message
        history.append(HumanMessage(content=query.strip()))
        
        # Generate response
        response = await self.llm.ainvoke(history)
        content = response.content
        
        # Add assistant response to history
        history.append(AIMessage(content=content))
        
        # Trim history to the most recent messages (preserving system prompt)
        if len(history) > self.max_history:
            new_history = []
            if self.system_prompt and history and isinstance(history[0], SystemMessage):
                new_history.append(history[0])
            tail_size = self.max_history - len(new_history)
            if tail_size > 0:
                new_history.extend(history[-tail_size:])
            history[:] = new_history
        
        return content, session_id