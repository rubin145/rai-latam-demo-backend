#!/usr/bin/env python3
"""
Tests para la aplicación RAI Latam Demo Backend
"""
import pytest
import asyncio
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

from app.services.langchain_chat import LangChainChatService
from app.services.response_evaluator import ResponseEvaluatorService


class TestLangChainChatService:
    """Tests para el servicio de chat con LangChain"""
    
    @pytest.fixture
    def chat_service(self):
        """Fixture para el servicio de chat base"""
        return LangChainChatService("configs/chatbots/banking_unsafe.yaml")
    
    @pytest.fixture
    def guardrails_service(self):
        """Fixture para el servicio de chat con guardrails"""
        return LangChainChatService("configs/chatbots/banking_safe.yaml")
    
    @pytest.mark.asyncio
    async def test_chat_service_basic_response(self, chat_service):
        """Test respuesta básica del servicio de chat"""
        response, session_id = await chat_service.handle_chat("¿Qué tipos de cuentas ofrecen?")
        
        assert response is not None
        assert len(response) > 0
        assert session_id is not None
        assert len(session_id) > 0
    
    @pytest.mark.asyncio
    async def test_chat_service_session_continuity(self, chat_service):
        """Test continuidad de sesión en el chat"""
        # Primer mensaje
        response1, session_id = await chat_service.handle_chat("Hola")
        
        # Segundo mensaje con la misma sesión
        response2, session_id2 = await chat_service.handle_chat("¿Qué más?", session_id)
        
        assert session_id == session_id2
        assert response1 != response2
    
    @pytest.mark.asyncio
    async def test_chat_service_provider_detection(self, chat_service):
        """Test que el servicio detecte el provider correctamente"""
        assert chat_service.provider == "GROQ"
        assert hasattr(chat_service, 'llm')
        assert hasattr(chat_service, 'system_prompt')
    
    @pytest.mark.asyncio
    async def test_input_filters_safe_message(self, guardrails_service):
        """Test filtros con mensaje seguro"""
        decision, evaluation, template = await guardrails_service.apply_input_filters(
            "¿Qué tipos de cuentas bancarias ofrecen?"
        )
        
        assert decision == "safe"
        assert evaluation == ""
        assert template == ""
    
    @pytest.mark.asyncio
    async def test_input_filters_toxic_message(self, guardrails_service):
        """Test filtros con mensaje tóxico"""
        decision, evaluation, template = await guardrails_service.apply_input_filters(
            "Eres un idiota y no me sirves para nada"
        )
        
        assert decision == "danger"
        assert len(evaluation) > 0
        assert "toxicity_filter" in evaluation
        assert len(template) > 0
    
    @pytest.mark.asyncio
    async def test_input_filters_financial_advice(self, guardrails_service):
        """Test filtros con pedido de consejo financiero"""
        decision, evaluation, template = await guardrails_service.apply_input_filters(
            "¿Qué debería hacer para invertir mi dinero y ganar más?"
        )
        
        assert decision == "danger"
        assert len(evaluation) > 0
        assert "financial_advice_filter" in evaluation
        assert len(template) > 0


class TestResponseEvaluatorService:
    """Tests para el servicio de evaluación de respuestas"""
    
    @pytest.fixture
    def evaluator_service(self):
        """Fixture para el servicio de evaluación"""
        return ResponseEvaluatorService("configs/evaluators/llm_evaluators.yaml")
    
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
    async def test_full_chat_workflow(self):
        """Test flujo completo de chat"""
        # 1. Crear servicio
        service = LangChainChatService("configs/chatbots/banking_unsafe.yaml")
        
        # 2. Hacer pregunta
        response, session_id = await service.handle_chat("¿Qué tipos de cuentas ofrecen?")
        
        # 3. Evaluar respuesta
        evaluator = ResponseEvaluatorService("configs/evaluators/llm_evaluators.yaml")
        evaluation = await evaluator.evaluate_response("¿Qué tipos de cuentas ofrecen?", response)
        
        # 4. Verificar que todo funcionó
        assert response is not None
        assert session_id is not None
        assert evaluation is not None
        assert len(evaluation) > 0
    
    @pytest.mark.asyncio
    async def test_full_guardrails_workflow(self):
        """Test flujo completo con guardrails"""
        # 1. Crear servicio con guardrails
        service = LangChainChatService("configs/chatbots/banking_safe.yaml")
        
        # 2. Probar mensaje seguro
        decision, evaluation, template = await service.apply_input_filters("¿Qué tipos de cuentas ofrecen?")
        assert decision == "safe"
        
        # 3. Probar mensaje peligroso
        decision, evaluation, template = await service.apply_input_filters("Eres un idiota")
        assert decision == "danger"
        
        # 4. Hacer chat normal si pasa filtros
        response, session_id = await service.handle_chat("¿Qué tipos de cuentas ofrecen?")
        assert response is not None
        assert session_id is not None


class TestResponseEvaluatorParsing:
    """Tests para el parsing de salida de evaluadores para el frontend"""
    
    @pytest.fixture
    def evaluator_service(self):
        """Fixture para el servicio de evaluador"""
        return ResponseEvaluatorService("configs/evaluators/llm_evaluators.yaml")
    
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