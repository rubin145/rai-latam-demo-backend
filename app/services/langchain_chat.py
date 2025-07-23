import uuid
import os
import asyncio
from typing import Tuple, Dict, Any

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langsmith import traceable, get_current_run_tree

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
            raise NotImplementedError("OpenAI provider not yet implemented")
        
        elif provider == "ANTHROPIC":
            # Future implementation
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
        """Apply input filters to query using parallel LCEL chains. Returns (decision, evaluation, template_response)"""
        input_filters = self.config.get("input_filters", [])
        
        # Create all filter tasks
        filter_tasks = [
            self._run_single_filter(filter_config, query) 
            for filter_config in input_filters
        ]
        
        # Wait for all filters to complete
        results = await asyncio.gather(*filter_tasks, return_exceptions=True)
        
        # Check if any filter rejected
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Filter {input_filters[i].get('name')} failed: {result}")
                continue
            if result and result[0] == "danger":
                return result
        
        return ("safe", "", "")
    
    def _create_filter_chain(self, filter_config: Dict[str, Any]):
        """Create a specialized LCEL chain for filter with specific configuration."""
        # Create filter-specific LLM
        filter_llm = self._create_filter_llm(filter_config)
        
        # Escape curly braces in system prompt to prevent variable interpretation
        system_prompt = filter_config["system_prompt"].replace("{", "{{").replace("}", "}}")
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}")
        ])
        
        # Create JSON output parser
        json_parser = JsonOutputParser()
        
        # Create and return the LCEL chain
        chain = (
            prompt
            | filter_llm
            | json_parser
        )
        
        return chain
    
    async def _run_single_filter(self, filter_config: Dict[str, Any], query: str) -> Tuple[str, str, str]:
        """Run a single filter and return its result."""
        try:
            filter_chain = self._create_filter_chain(filter_config)
            filter_result = await filter_chain.ainvoke({"query": query.strip()})
            
            return (
                filter_result.get("decision", "safe"),
                f"{filter_config.get('name', 'filter')}: {filter_result.get('evaluation', '')}",
                filter_config.get("template_response", "Sorry, I can't help with that.")
            )
        except Exception as e:
            # Return safe on filter failure
            return ("safe", f"Filter error: {str(e)}", "")
    
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
    
    @traceable(name="chat_with_filters")
    async def handle_chat(self, query: str, session_id: str = None, callbacks=None) -> Tuple[str, str]:
        """Handle a chat message and return response with session ID."""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Set session_id in trace metadata for threading
        if run_tree := get_current_run_tree():
            run_tree.extra = run_tree.extra or {}
            run_tree.extra["metadata"] = {"session_id": session_id}
        
        # Apply filters if configured
        filter_result = await self.apply_input_filters(query)
        if filter_result[0] == "danger":
            return filter_result[2], session_id  # Return rejection message
        
        # Get or create history for this session
        history = self.chat_history.setdefault(session_id, [])
        
        # Add system prompt if not present and configured
        if self.system_prompt and not any(isinstance(msg, SystemMessage) for msg in history):
            history.insert(0, SystemMessage(content=self.system_prompt))
        
        # Add user message
        history.append(HumanMessage(content=query.strip()))
        
        # Generate response with callbacks for evaluation
        response = await self.llm.ainvoke(history, callbacks=callbacks or [])
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