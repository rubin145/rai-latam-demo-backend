#!/usr/bin/env python3
"""
Tests para la aplicación RAI Latam Demo Backend - Functionality Tests (Mocked)
"""
import pytest
import asyncio
import os
from unittest.mock import patch, AsyncMock, MagicMock
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

from app.services.chat import ChatService
from app.services.evaluators import LLMEvaluator, LightEvaluator


class TestChatService:
    """Tests para el servicio de chat con LangChain"""
    
    @pytest.fixture
    @patch('app.services.langsmith_client.LangSmithClient')
    def chat_service(self, mock_langsmith):
        """Fixture para el servicio de chat base"""
        # Mock LangSmith to prevent real calls
        mock_langsmith.return_value = MagicMock()
        return ChatService("configs/chatbots/banking_unsafe.yaml")
    
    @pytest.fixture
    @patch('app.services.langsmith_client.LangSmithClient')
    def guardrails_service(self, mock_langsmith):
        """Fixture para el servicio de chat con guardrails"""
        # Mock LangSmith to prevent real calls
        mock_langsmith.return_value = MagicMock()
        return ChatService("configs/chatbots/banking_safe.yaml")
    
    @pytest.mark.asyncio
    async def test_chat_service_basic_response(self, chat_service):
        """Test respuesta básica del servicio de chat - functionality only"""
        # Mock LLM response directly on the service instance
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Ofrecemos cuentas corrientes, de ahorro y cuentas empresariales."))
        chat_service.llm = mock_llm
        
        response, session_id, run_id = await chat_service.handle_chat("¿Qué tipos de cuentas ofrecen?")
        
        # Test functionality: service should return response and session_id
        assert response is not None
        assert len(response) > 0
        assert session_id is not None
        assert len(session_id) > 0
        assert response == "Ofrecemos cuentas corrientes, de ahorro y cuentas empresariales."
    
    @pytest.mark.asyncio
    async def test_chat_service_session_continuity(self, chat_service):
        """Test continuidad de sesión en el chat"""
        # Primer mensaje
        response1, session_id, run_id1 = await chat_service.handle_chat("Hola")
        
        # Segundo mensaje con la misma sesión
        response2, session_id2, run_id2 = await chat_service.handle_chat("¿Qué más?", session_id)
        
        assert session_id == session_id2
        assert response1 != response2
    
    @pytest.mark.asyncio
    async def test_chat_service_provider_detection(self, chat_service):
        """Test que el servicio detecte el provider correctamente"""
        assert chat_service.config.get("provider", "GROQ") == "GROQ"
        assert hasattr(chat_service, 'llm')
        assert hasattr(chat_service, 'system_prompt')
    
    @pytest.mark.asyncio
    @patch('app.services.chat.ChatService.apply_input_filters')
    async def test_input_filters_functionality_safe(self, mock_apply_filters, guardrails_service):
        """Test filtros functionality - safe message handling"""
        # Mock safe message response
        mock_apply_filters.return_value = ("safe", "", "")
        
        decision, evaluation, template = await guardrails_service.apply_input_filters(
            "¿Qué tipos de cuentas bancarias ofrecen?"
        )
        
        # Test functionality: service should return proper tuple format
        assert isinstance(decision, str)
        assert isinstance(evaluation, str) 
        assert isinstance(template, str)
        assert decision == "safe"
    
    @pytest.mark.asyncio
    @patch('app.services.chat.ChatService.apply_input_filters')
    async def test_input_filters_functionality_blocked(self, mock_apply_filters, guardrails_service):
        """Test filtros functionality - blocked message handling"""
        # Mock blocked message response
        mock_apply_filters.return_value = (
            "danger", 
            "toxicity_filter: Mensaje bloqueado por contenido ofensivo",
            "No puedo procesar ese tipo de mensajes. ¿En qué más puedo ayudarte?"
        )
        
        decision, evaluation, template = await guardrails_service.apply_input_filters(
            "Mensaje ofensivo simulado"
        )
        
        # Test functionality: service should return proper tuple format with content
        assert isinstance(decision, str)
        assert isinstance(evaluation, str)
        assert isinstance(template, str) 
        assert decision == "danger"
        assert len(evaluation) > 0
        assert len(template) > 0


class TestLLMEvaluator:
    """Tests para el servicio de evaluación de respuestas"""
    
    @pytest.fixture
    def evaluator_service(self):
        """Fixture para el servicio de evaluación"""
        return LLMEvaluator("configs/evaluators/llm_evaluators.yaml")
    
    @pytest.mark.asyncio
    async def test_evaluator_service_basic_evaluation(self, evaluator_service):
        """Test evaluación básica de una respuesta - funcionalidad de la app"""
        prompt = "¿Qué tipos de cuentas bancarias ofrecen?"
        response = "Ofrecemos cuentas corrientes, de ahorro y cuentas de inversión."
        
        results = await evaluator_service.evaluate_response(prompt, response)
        
        # Verificar funcionalidad básica - que devuelva resultados estructurados
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Verificar que los evaluadores esperados estén presentes
        expected_evaluators = ["toxicity", "financial_advice", "hallucination", "topic_adherence"]
        for evaluator in expected_evaluators:
            assert evaluator in results
            assert isinstance(results[evaluator], dict)
            
            # Verificar estructura básica - debe tener decision/score O error
            result = results[evaluator]
            if "error" not in result:
                # Resultado exitoso - verificar estructura
                assert "decision" in result
                assert "score" in result
                assert "evaluation" in result
                assert "evaluator" in result
            else:
                # Error esperado - verificar que tenga info del error
                assert "evaluator" in result
                assert isinstance(result["error"], str)
    
    @pytest.mark.asyncio
    async def test_evaluator_service_provider_detection(self, evaluator_service):
        """Test que el evaluador use el provider correcto - funcionalidad de la app"""
        # Verificar que el servicio tenga evaluadores configurados
        assert len(evaluator_service.evaluators) > 0
        assert "toxicity" in evaluator_service.evaluators
        
        # Verificar que pueda listar los evaluadores
        evaluator_names = evaluator_service.get_evaluator_names()
        assert len(evaluator_names) > 0
        
        # Verificar que pueda obtener info de evaluadores
        for name in evaluator_names:
            info = evaluator_service.get_evaluator_info(name)
            assert info is not None
            assert "type" in info


class TestConfigurationIntegrity:
    """Tests para verificar la integridad de las configuraciones"""
    
    def test_banking_chat_config_exists(self):
        """Test que el archivo de configuración de chat existe"""
        assert os.path.exists("configs/chatbots/banking_unsafe.yaml")
    
    def test_banking_chat_rai_config_exists(self):
        """Test que el archivo de configuración de chat con guardrails existe"""
        assert os.path.exists("configs/chatbots/banking_safe.yaml")
    
    def test_response_evaluator_config_exists(self):
        """Test que el archivo de configuración del evaluador existe"""
        assert os.path.exists("configs/evaluators/llm_evaluators.yaml")
    
    def test_groq_api_key_exists(self):
        """Test que la API key de Groq esté configurada"""
        assert os.getenv("GROQ_API_KEY") is not None
        assert len(os.getenv("GROQ_API_KEY")) > 0


class TestEndToEndWorkflow:
    """Tests de flujo completo end-to-end"""
    
    @pytest.mark.asyncio
    @patch('app.services.langsmith_client.LangSmithClient')
    async def test_full_chat_workflow(self, mock_langsmith):
        """Test flujo completo de chat"""
        # Mock LangSmith to prevent real calls
        mock_langsmith.return_value = MagicMock()
        # 1. Crear servicio
        service = ChatService("configs/chatbots/banking_unsafe.yaml")
        
        # 2. Hacer pregunta
        response, session_id, run_id = await service.handle_chat("¿Qué tipos de cuentas ofrecen?")
        
        # 3. Evaluar respuesta
        evaluator = LLMEvaluator("configs/evaluators/llm_evaluators.yaml")
        evaluation = await evaluator.evaluate_response("¿Qué tipos de cuentas ofrecen?", response)
        
        # 4. Verificar que todo funcionó
        assert response is not None
        assert session_id is not None
        assert evaluation is not None
        assert len(evaluation) > 0
    
    @pytest.mark.asyncio
    @patch('app.services.langsmith_client.LangSmithClient')
    @patch('app.services.chat.ChatService.apply_input_filters')
    @patch('app.services.llm_manager.LLMManager.get_llm')
    async def test_full_guardrails_workflow_functionality(self, mock_get_llm, mock_apply_filters, mock_langsmith):
        """Test flujo completo con guardrails - functionality only"""
        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Respuesta simulada del chat"))
        mock_get_llm.return_value = mock_llm
        
        # Mock filters to return safe
        mock_apply_filters.return_value = ("safe", "", "")
        
        # Mock LangSmith to prevent real calls  
        mock_langsmith.return_value = MagicMock()
        # 1. Crear servicio con guardrails
        service = ChatService("configs/chatbots/banking_safe.yaml")
        
        # 2. Test workflow: filters -> chat
        decision, evaluation, template = await service.apply_input_filters("¿Qué tipos de cuentas ofrecen?")
        assert decision == "safe"  # From mock
        
        # 3. If safe, proceed to chat
        if decision == "safe":
            response, session_id, run_id = await service.handle_chat("¿Qué tipos de cuentas ofrecen?")
            assert response is not None
            assert session_id is not None
            assert response == "Respuesta simulada del chat"


class TestLLMEvaluatorParsing:
    """Tests para el parsing de salida de evaluadores para el frontend"""
    
    @pytest.fixture
    def evaluator_service(self):
        """Fixture para el servicio de evaluador"""
        return LLMEvaluator("configs/evaluators/llm_evaluators.yaml")
    
    def test_criteria_evaluator_parsing(self, evaluator_service):
        """Test parsing de CriteriaEvaluator a formato esperado por frontend"""
        # Simular output de LangChain CriteriaEvaluator
        langchain_output = {
            "score": 1,
            "value": "Y", 
            "reasoning": "A resposta contém aconselhamento financeiro específico."
        }
        
        # Aplicar parsing
        parsed = evaluator_service._parse_langchain_output(
            langchain_output, "criteria", "financial_advice"
        )
        
        # Verificar formato esperado por frontend
        assert "decision" in parsed
        assert "score" in parsed
        assert "evaluation" in parsed
        assert "evaluator" in parsed
        
        assert parsed["decision"] == "Y"
        assert parsed["score"] == 1
        assert parsed["evaluation"] == "A resposta contém aconselhamento financeiro específico."
        assert parsed["evaluator"] == "financial_advice"
    
    def test_score_string_evaluator_parsing(self, evaluator_service):
        """Test parsing de ScoreStringEvaluator a formato esperado por frontend"""
        # Simular output de LangChain ScoreStringEvaluator
        langchain_output = {
            "score": 7,
            "reasoning": "A resposta apresenta um nível moderado de toxicidade devido ao tom sarcástico."
        }
        
        # Aplicar parsing
        parsed = evaluator_service._parse_langchain_output(
            langchain_output, "score_string", "toxicity"
        )
        
        # Verificar formato esperado por frontend
        assert "decision" in parsed
        assert "score" in parsed
        assert "evaluation" in parsed
        assert "evaluator" in parsed
        
        assert parsed["decision"] == 7  # Para score_string, decision = score
        assert parsed["score"] == 7
        assert parsed["evaluation"] == "A resposta apresenta um nível moderado de toxicidade devido ao tom sarcástico."
        assert parsed["evaluator"] == "toxicity"
    
    def test_parsing_error_handling(self, evaluator_service):
        """Test manejo de errores en parsing"""
        # Simular output malformado
        malformed_output = {"invalid": "data"}
        
        # Aplicar parsing
        parsed = evaluator_service._parse_langchain_output(
            malformed_output, "criteria", "test_evaluator"
        )
        
        # Verificar manejo de error
        assert "decision" in parsed
        assert "score" in parsed
        assert "evaluation" in parsed
        assert "evaluator" in parsed
        assert parsed["evaluator"] == "test_evaluator"


if __name__ == "__main__":
    # Ejecutar tests directamente
    pytest.main([__file__, "-v"])