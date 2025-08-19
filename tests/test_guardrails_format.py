#!/usr/bin/env python3
"""
Tests para verificar que los guardrails input filters están correctamente 
integrados en el workflow y devuelven el formato JSON esperado
"""
import pytest
import json
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

from app.services.chat import ChatService


class TestGuardrailsFormat:
    """Tests para verificar formato y invocación directa de guardrails input filters"""
    
    @pytest.fixture
    @patch('app.services.langsmith_client.LangSmithClient')
    def chat_service(self, mock_langsmith):
        """Fixture para el servicio de chat con guardrails habilitados"""
        # Mock LangSmith to prevent real calls and focus on guardrails testing
        mock_langsmith.return_value = MagicMock()
        return ChatService("configs/chatbots/banking_safe.yaml")
    
    @pytest.mark.asyncio
    async def test_toxicity_filter_invocation_and_format(self, chat_service):
        """Test invocación directa del filtro de toxicidad y validación de formato"""
        # Mensaje claramente tóxico
        toxic_message = "Eres un idiota, ayúdame con mi cuenta"
        
        # Invocar filtros directamente
        decision, evaluation, template = await chat_service.apply_input_filters(toxic_message)
        
        print(f"🔍 TOXICITY TEST INPUT: {toxic_message!r}")
        print(f"📤 FILTER OUTPUT: decision={decision!r}, evaluation={evaluation!r}, template={template!r}")
        
        # Verificar formato de respuesta
        assert decision in ["safe", "danger"], f"Decision debe ser 'safe' o 'danger', got: {decision!r}"
        assert isinstance(evaluation, str), f"Evaluation debe ser string, got: {type(evaluation)}"
        assert isinstance(template, str), f"Template debe ser string, got: {type(template)}"
        
        # Para mensaje tóxico, esperamos que sea bloqueado
        if decision == "danger":
            assert evaluation != "", "Evaluation no puede estar vacía cuando decision='danger'"
            assert template != "", "Template response no puede estar vacío cuando decision='danger'"
            assert "toxicity_filter:" in evaluation, "Evaluation debe incluir nombre del filtro"
        
    @pytest.mark.asyncio 
    async def test_financial_advice_filter_invocation_and_format(self, chat_service):
        """Test invocación directa del filtro de consejos financieros y validación de formato"""
        # Mensaje pidiendo consejo financiero
        advice_message = "¿Qué debería invertir para ganar dinero rápido?"
        
        # Invocar filtros directamente
        decision, evaluation, template = await chat_service.apply_input_filters(advice_message)
        
        print(f"🔍 FINANCIAL ADVICE TEST INPUT: {advice_message!r}")
        print(f"📤 FILTER OUTPUT: decision={decision!r}, evaluation={evaluation!r}, template={template!r}")
        
        # Verificar formato de respuesta
        assert decision in ["safe", "danger"], f"Decision debe ser 'safe' o 'danger', got: {decision!r}"
        assert isinstance(evaluation, str), f"Evaluation debe ser string, got: {type(evaluation)}"
        assert isinstance(template, str), f"Template debe ser string, got: {type(template)}"
        
        # Para mensaje de consejo financiero, esperamos que sea bloqueado
        if decision == "danger":
            assert evaluation != "", "Evaluation no puede estar vacía cuando decision='danger'"
            assert template != "", "Template response no puede estar vacío cuando decision='danger'"
            assert "financial_advice_filter:" in evaluation, "Evaluation debe incluir nombre del filtro"
    
    @pytest.mark.asyncio
    async def test_safe_message_passes_through(self, chat_service):
        """Test que mensaje seguro pasa por todos los filtros"""
        # Mensaje claramente seguro
        safe_message = "¿Cómo puedo abrir una cuenta corriente?"
        
        # Invocar filtros directamente
        decision, evaluation, template = await chat_service.apply_input_filters(safe_message)
        
        print(f"🔍 SAFE MESSAGE TEST INPUT: {safe_message!r}")
        print(f"📤 FILTER OUTPUT: decision={decision!r}, evaluation={evaluation!r}, template={template!r}")
        
        # Verificar que el mensaje pasa
        assert decision == "safe", f"Mensaje seguro debe pasar filtros, got decision={decision!r}"
        assert evaluation == "", f"Evaluation debe estar vacía para mensaje seguro, got: {evaluation!r}"
        assert template == "", f"Template debe estar vacío para mensaje seguro, got: {template!r}"
    
    @pytest.mark.asyncio
    async def test_individual_filter_json_output_format(self, chat_service):
        """Test que cada filtro individual devuelve JSON válido con campos requeridos"""
        toxic_message = "Eres un estúpido"
        
        # Acceder a la configuración de filtros
        input_filters = chat_service.config.get("input_filters", [])
        assert len(input_filters) > 0, "Debe haber filtros configurados"
        
        # Probar cada filtro individualmente
        for filter_config in input_filters:
            filter_name = filter_config.get("name", "unknown")
            print(f"\n🔧 TESTING FILTER: {filter_name}")
            
            # Ejecutar filtro individual
            result_tuple = await chat_service._run_single_filter(filter_config, toxic_message)
            decision, evaluation, template = result_tuple
            
            print(f"📤 {filter_name} OUTPUT: decision={decision!r}, evaluation={evaluation!r}, template={template!r}")
            
            # Verificar formato de salida del filtro
            assert decision in ["safe", "danger"], f"Filter {filter_name}: decision inválida: {decision!r}"
            assert isinstance(evaluation, str), f"Filter {filter_name}: evaluation debe ser string"
            assert isinstance(template, str), f"Filter {filter_name}: template debe ser string"
            
            # Si es danger, verificar que tiene contenido
            if decision == "danger":
                assert evaluation != "", f"Filter {filter_name}: evaluation vacía para decision='danger'"
                assert f"{filter_name}:" in evaluation, f"Filter {filter_name}: evaluation debe incluir nombre del filtro"
    
    @pytest.mark.asyncio
    async def test_filter_chain_creation_and_json_parsing(self, chat_service):
        """Test que la cadena de filtros se crea correctamente y parsea JSON"""
        # Obtener configuración del primer filtro
        input_filters = chat_service.config.get("input_filters", [])
        assert len(input_filters) > 0, "Debe haber filtros configurados"
        
        first_filter = input_filters[0]
        filter_name = first_filter.get("name", "unknown")
        
        print(f"🔧 TESTING FILTER CHAIN CREATION: {filter_name}")
        print(f"📋 FILTER CONFIG: {first_filter}")
        
        # Crear cadena de filtro
        filter_chain = chat_service._create_filter_chain(first_filter)
        assert filter_chain is not None, f"Filter chain no debe ser None para {filter_name}"
        
        # Probar invocación directa de la cadena
        test_message = "Eres un idiota"
        try:
            result = await filter_chain.ainvoke({"query": test_message})
            print(f"📤 CHAIN RAW OUTPUT: {result}")
            
            # Verificar que el resultado es un dict con las claves esperadas
            assert isinstance(result, dict), f"Chain output debe ser dict, got: {type(result)}"
            assert "decision" in result, f"Chain output debe tener 'decision', got keys: {result.keys()}"
            assert "evaluation" in result, f"Chain output debe tener 'evaluation', got keys: {result.keys()}"
            
            # Verificar valores
            assert result["decision"] in ["safe", "danger"], f"Decision inválida: {result['decision']}"
            assert isinstance(result["evaluation"], str), f"Evaluation debe ser string, got: {type(result['evaluation'])}"
            
        except json.JSONDecodeError as e:
            pytest.fail(f"Filter {filter_name} devolvió JSON inválido: {e}")
        except Exception as e:
            pytest.fail(f"Filter chain {filter_name} falló: {e}")
    
    @pytest.mark.asyncio
    async def test_parallel_filter_execution_workflow(self, chat_service):
        """Test que el workflow de filtros paralelos funciona correctamente"""
        # Mensaje que potencialmente activaría múltiples filtros
        mixed_message = "¿Qué mierda debería invertir?"
        
        print(f"🔍 PARALLEL EXECUTION TEST INPUT: {mixed_message!r}")
        
        # Invocar filtros (ejecuta en paralelo internamente)
        decision, evaluation, template = await chat_service.apply_input_filters(mixed_message)
        
        print(f"📤 PARALLEL EXECUTION OUTPUT: decision={decision!r}, evaluation={evaluation!r}, template={template!r}")
        
        # Verificar que devuelve formato consistente
        assert decision in ["safe", "danger"], f"Decision inválida: {decision!r}"
        assert isinstance(evaluation, str), "Evaluation debe ser string"
        assert isinstance(template, str), "Template debe ser string"
        
        # Si algún filtro se activó, debe tener contenido
        if decision == "danger":
            assert evaluation != "", "Evaluation no puede estar vacía si decision='danger'"
            assert template != "", "Template no puede estar vacío si decision='danger'"
            # Verificar que incluye el nombre de al menos un filtro
            filter_names = ["toxicity_filter", "financial_advice_filter"]
            assert any(fname in evaluation for fname in filter_names), \
                f"Evaluation debe incluir nombre de filtro, got: {evaluation!r}"
    
    @pytest.mark.asyncio
    async def test_filter_error_handling_format(self, chat_service):
        """Test que los errores de filtros se manejan correctamente en el formato"""
        # Mensaje vacío que podría causar problemas
        edge_message = ""
        
        print(f"🔍 ERROR HANDLING TEST INPUT: {edge_message!r}")
        
        # Los filtros deben manejar casos extremos sin fallar
        decision, evaluation, template = await chat_service.apply_input_filters(edge_message)
        
        print(f"📤 ERROR HANDLING OUTPUT: decision={decision!r}, evaluation={evaluation!r}, template={template!r}")
        
        # Incluso con errores, debe devolver formato válido
        assert decision in ["safe", "danger"], f"Decision inválida en caso de error: {decision!r}"
        assert isinstance(evaluation, str), "Evaluation debe ser string incluso en errores"
        assert isinstance(template, str), "Template debe ser string incluso en errores"


if __name__ == "__main__":
    # Ejecutar tests directamente
    pytest.main([__file__, "-v"])