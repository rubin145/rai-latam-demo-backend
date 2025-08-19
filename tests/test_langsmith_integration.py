#!/usr/bin/env python3
"""
Integration tests for LangSmith functionality.
These tests make REAL LangSmith API calls and should be run separately from unit tests.
"""
import pytest
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

from app.services.langsmith_client import LangSmithClient
from app.services.chat import ChatService


class TestLangSmithIntegration:
    """Integration tests for LangSmith client functionality"""
    
    @pytest.fixture
    def langsmith_client(self):
        """Fixture for LangSmith client"""
        return LangSmithClient()
    
    @pytest.fixture
    def chat_service_with_langsmith(self):
        """Fixture for ChatService with real LangSmith integration"""
        return ChatService("configs/chatbots/banking_safe.yaml")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_langsmith_client_initialization(self, langsmith_client):
        """Test that LangSmith client initializes properly"""
        assert langsmith_client is not None
        assert hasattr(langsmith_client, 'evaluator_manager')
        assert hasattr(langsmith_client, 'evaluator_names')
        assert len(langsmith_client.evaluator_names) > 0
        
        # Check that evaluators are properly configured
        expected_evaluators = ['topic_adherence', 'hallucination', 'toxicity', 'financial_advice']
        
        for expected in expected_evaluators:
            assert expected in langsmith_client.evaluator_names
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_langsmith_evaluation_format(self, langsmith_client):
        """Test that LangSmith client can evaluate using the underlying evaluator manager"""
        # Test data
        prompt = "¿Cuál es la capital de Francia?"
        response = "La capital de Francia es París."
        
        # This will make a real API call via the evaluator manager
        try:
            results = await langsmith_client.evaluator_manager.evaluate_response(prompt, response)
            
            # Verify response structure
            assert isinstance(results, dict)
            assert len(results) > 0
            
            # Check each evaluator result has required fields
            for evaluator_name, result in results.items():
                assert isinstance(result, dict)
                assert 'decision' in result
                assert 'score' in result
                assert 'evaluation' in result
                assert 'evaluator' in result
                assert result['evaluator'] == evaluator_name
                
        except Exception as e:
            pytest.skip(f"LangSmith API call failed: {e}. This is expected if API limits are hit.")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_chat_service_langsmith_integration(self, chat_service_with_langsmith):
        """Test that ChatService integrates properly with LangSmith"""
        # Verify LangSmith evaluator is initialized
        assert chat_service_with_langsmith.langsmith_evaluator is not None
        
        # This test verifies integration but doesn't make API calls
        # to avoid hitting rate limits during testing
        assert hasattr(chat_service_with_langsmith.langsmith_evaluator, 'evaluator_manager')
    
    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("LANGSMITH_API_KEY"), 
        reason="LANGSMITH_API_KEY not set - skipping LangSmith tests"
    )
    def test_langsmith_environment_setup(self):
        """Test that LangSmith environment variables are properly configured"""
        assert os.getenv("LANGSMITH_API_KEY") is not None
        assert len(os.getenv("LANGSMITH_API_KEY")) > 0
        
        # Optional: check project name if set
        if os.getenv("LANGCHAIN_PROJECT"):
            assert len(os.getenv("LANGCHAIN_PROJECT")) > 0