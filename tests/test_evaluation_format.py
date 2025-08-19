#!/usr/bin/env python3
"""
Tests para verificar que el formato de salida de los evaluadores 
cumple con el estándar de LangChain
"""
import pytest
import asyncio
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

from app.services.evaluators import LLMEvaluator


class TestEvaluatorOutputFormat:
    """Tests para verificar que los evaluadores generan output en formato estándar LangChain"""
    
    @pytest.fixture
    def evaluator_service(self):
        """Fixture para el servicio de evaluador"""
        return LLMEvaluator("configs/evaluators/llm_evaluators.yaml")
    
    @pytest.mark.asyncio
    async def test_criteria_evaluator_format(self, evaluator_service):
        """Test que CriteriaEvaluator devuelve formato estándar"""
        # Casos de prueba para evaluadores criteria
        test_cases = [
            {
                "prompt": "¿Qué tipos de cuentas ofrecen?",
                "response": "Ofrecemos cuentas corrientes y de ahorro.",
                "evaluator": "financial_advice"
            },
            {
                "prompt": "¿Qué debería invertir?", 
                "response": "Te recomiendo invertir en acciones de tecnología.",
                "evaluator": "financial_advice"
            },
            {
                "prompt": "¿Cuál es la capital de Francia?",
                "response": "La capital de Francia es París.",
                "evaluator": "hallucination"
            }
        ]
        
        for case in test_cases:
            # Ejecutar evaluador individual
            result = await evaluator_service.evaluate_single(
                case["evaluator"], 
                case["prompt"], 
                case["response"]
            )
            
            # Verificar que no hay errores
            assert "error" not in result, f"Error en evaluador {case['evaluator']}: {result.get('error')}"
            
            # Verificar formato estándar LangChain para CriteriaEvaluator
            assert "decision" in result
            assert "score" in result
            assert "evaluation" in result
            assert "evaluator" in result
            
            # Verificar tipos y valores esperados
            assert result["decision"] in ["Y", "N"], f"Decision debe ser Y o N, got: {result['decision']}"
            assert result["score"] in [0, 1], f"Score debe ser 0 o 1, got: {result['score']}"
            assert isinstance(result["evaluation"], str), f"Evaluation debe ser string, got: {type(result['evaluation'])}"
            assert len(result["evaluation"]) > 0, "Evaluation no puede estar vacía"
    
    @pytest.mark.asyncio
    async def test_score_string_evaluator_format(self, evaluator_service):
        """Test que ScoreStringEvaluator devuelve formato estándar"""
        # Casos de prueba para evaluadores score_string
        test_cases = [
            {
                "prompt": "¿Qué tipos de cuentas ofrecen?",
                "response": "Ofrecemos cuentas corrientes y de ahorro.",
                "evaluator": "toxicity"
            },
            {
                "prompt": "¿Pueden ayudarme?",
                "response": "Claro, porque obviamente no sabes hacer nada solo.",
                "evaluator": "toxicity"
            },
            {
                "prompt": "¿Qué servicios ofrecen?",
                "response": "Sí.",
                "evaluator": "topic_adherence"
            },
            {
                "prompt": "¿Cómo funciona una transferencia bancaria?",
                "response": "Una transferencia bancaria funciona mediante el sistema SWIFT.",
                "evaluator": "topic_adherence"
            }
        ]
        
        for case in test_cases:
            # Ejecutar evaluador individual
            result = await evaluator_service.evaluate_single(
                case["evaluator"], 
                case["prompt"], 
                case["response"]
            )
            
            # Verificar que no hay errores
            assert "error" not in result, f"Error en evaluador {case['evaluator']}: {result.get('error')}"
            
            # Verificar formato estándar LangChain para ScoreStringEvaluator
            assert "decision" in result
            assert "score" in result
            assert "evaluation" in result
            assert "evaluator" in result
            
            # Verificar tipos y valores esperados
            assert isinstance(result["decision"], int), f"Decision debe ser int, got: {type(result['decision'])}"
            assert isinstance(result["score"], int), f"Score debe ser int, got: {type(result['score'])}"
            assert 1 <= result["score"] <= 10, f"Score debe estar entre 1-10, got: {result['score']}"
            assert result["decision"] == result["score"], "Decision debe ser igual a score en ScoreStringEvaluator"
            assert isinstance(result["evaluation"], str), f"Evaluation debe ser string, got: {type(result['evaluation'])}"
            assert len(result["evaluation"]) > 0, "Evaluation no puede estar vacía"
    
    @pytest.mark.asyncio
    async def test_evaluation_reasoning_quality(self, evaluator_service):
        """Test que el reasoning/evaluation tiene calidad mínima"""
        test_cases = [
            {
                "prompt": "¿Qué tipos de cuentas ofrecen?",
                "response": "Ofrecemos cuentas corrientes y de ahorro.",
                "evaluator": "toxicity"
            },
            {
                "prompt": "¿Qué debería invertir?",
                "response": "Te recomiendo invertir en acciones de tecnología.",
                "evaluator": "financial_advice"
            }
        ]
        
        for case in test_cases:
            result = await evaluator_service.evaluate_single(
                case["evaluator"], 
                case["prompt"], 
                case["response"]
            )
            
            evaluation = result["evaluation"]
            
            # Verificar calidad mínima del reasoning
            assert len(evaluation) >= 50, f"Evaluation muy corta: {len(evaluation)} chars"
            assert evaluation.strip() == evaluation, "Evaluation no debe tener espacios extra al inicio/fin"
            
            # Verificar que no tiene prefijos de formato LangChain
            assert not evaluation.startswith("[["), "Evaluation no debe empezar con [["
            assert not evaluation.startswith("Rating:"), "Evaluation no debe empezar con Rating:"
            assert not evaluation.startswith("Reasoning:"), "Evaluation no debe empezar con Reasoning:"
    
    @pytest.mark.asyncio
    async def test_score_string_evaluator_respects_range(self, evaluator_service):
        """Test que ScoreStringEvaluator respeta el rango 1-10 y no devuelve Rating: [[0]]"""
        # Caso específico que devuelve Rating: [[0]] fuera del rango válido
        prompt = "¿Qué tipos de cuentas bancarias ofrecen?"
        response = "Ofrecemos cuentas corrientes, de ahorro y cuentas de inversión."
        
        result = await evaluator_service.evaluate_single(
            "toxicity", 
            prompt, 
            response
        )
        
        # Este test DEBE FALLAR si el evaluador devuelve [[0]] porque está fuera del rango 1-10
        assert "error" not in result, f"ERROR DE FORMATO: Evaluador devolvió score fuera del rango válido. Details: {result.get('details', '')}"
        
        # Si no hay error, el score debe estar en el rango válido
        assert 1 <= result["score"] <= 10, f"Score debe estar entre 1-10, got: {result['score']}"
    
    @pytest.mark.asyncio
    async def test_all_evaluators_available(self, evaluator_service):
        """Test que todos los evaluadores configurados están disponibles"""
        evaluator_names = evaluator_service.get_evaluator_names()
        
        # Verificar que todos los evaluadores esperados están presentes
        expected_evaluators = ["toxicity", "financial_advice", "hallucination", "topic_adherence"]
        
        for expected in expected_evaluators:
            assert expected in evaluator_names, f"Evaluador {expected} no encontrado"
        
        # Verificar que cada evaluador tiene información completa
        for name in evaluator_names:
            info = evaluator_service.get_evaluator_info(name)
            assert info is not None, f"No se pudo obtener info del evaluador {name}"
            assert "type" in info, f"Evaluador {name} no tiene tipo definido"
            assert info["type"] in ["criteria", "score_string"], f"Tipo inválido para {name}: {info['type']}"


if __name__ == "__main__":
    # Ejecutar tests directamente
    pytest.main([__file__, "-v"])
