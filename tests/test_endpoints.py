import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

# Mock LangSmith before importing the app to prevent real calls during FastAPI initialization
with patch('app.services.langsmith_client.LangSmithClient') as mock_langsmith:
    mock_langsmith.return_value = MagicMock()
    from app.main import app  # Asume que tu instancia de FastAPI se llama 'app' en 'app/main.py'

client = TestClient(app)

def test_read_root():
    """
    Test para el endpoint raíz (/).
    Verifica que la API esté en funcionamiento y devuelva un mensaje de bienvenida.
    """
    response = client.get("/") # Ruta corregida a "/"
    assert response.status_code == 200
    # Assert para el JSON actual que devuelve la API
    expected_json = {"message": "RAI Demo API", "version": "1.0.0", "docs": "/api/docs"}
    assert response.json() == expected_json

@patch('app.services.evaluators.LLMEvaluator.evaluate_response', new_callable=AsyncMock)
def test_evaluate_response_endpoint_success(mock_evaluate):
    """
    Test para un caso exitoso del endpoint /api/evaluate_response.
    Tests API functionality with mocked evaluator response.
    """
    # Mock evaluator response with expected format
    mock_evaluate.return_value = {
        "toxicity": {
            "decision": 1,
            "score": 1,
            "evaluation": "La respuesta no contiene contenido tóxico",
            "evaluator": "toxicity"
        },
        "hallucination": {
            "decision": "N", 
            "score": 0,
            "evaluation": "La información sobre París como capital de Francia es correcta",
            "evaluator": "hallucination"
        }
    }
    
    payload = {
        "prompt": "¿Cuál es la capital de Francia?",
        "response": "La capital de Francia es París."
    }
    response = client.post("/api/evaluate", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    # El schema de respuesta contiene 'results' que es un diccionario
    assert "results" in data
    assert isinstance(data["results"], dict)
    assert len(data["results"]) == 2  # toxicity and hallucination
    
    # Verificar estructura de respuesta de toxicity
    toxicity_result = data["results"]["toxicity"]
    assert "evaluator" in toxicity_result
    assert "score" in toxicity_result
    assert "evaluation" in toxicity_result
    assert toxicity_result["evaluator"] == "toxicity"
    assert isinstance(toxicity_result["score"], (int, float))
    
    # Verificar estructura de respuesta de hallucination
    hallucination_result = data["results"]["hallucination"]
    assert "evaluator" in hallucination_result
    assert "score" in hallucination_result  
    assert "evaluation" in hallucination_result
    assert hallucination_result["evaluator"] == "hallucination"

def test_evaluate_response_endpoint_missing_fields():
    """
    Test para el endpoint /api/evaluate_response cuando faltan campos en el payload.
    Espera una respuesta de error 422 (Unprocessable Entity) de FastAPI.
    """
    payload = {
        "prompt": "Solo un campo, falta la respuesta"
    }
    response = client.post("/api/evaluate", json=payload) # Ruta corregida
    
    assert response.status_code == 422 # Error de validación de Pydantic

# --- Tests para los endpoints de Chat ---

@patch('app.services.chat.ChatService.apply_input_filters', new_callable=AsyncMock)
@patch('app.services.chat.ChatService.handle_chat', new_callable=AsyncMock)
def test_chat_endpoint_success(mock_handle_chat, mock_apply_filters):
    """
    Test para un caso exitoso del endpoint /api/chatbots/{chatbot_id}/chat.
    Tests API functionality with mocked guardrails response.
    """
    # Mock guardrails response (safe message)
    mock_apply_filters.return_value = ("safe", "", "")
    
    # Mock chat response
    mock_handle_chat.return_value = ("Hola, estoy bien. ¿En qué puedo ayudarte?", "session_123", "run_123")
    
    payload = {"query": "Hola, ¿cómo estás?"}
    response = client.post("/api/chatbots/banking_safe/chat?use_guardrails=true", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "response" in data
    assert "session_id" in data
    assert "run_id" in data
    assert "filter_triggered" in data
    assert data["filter_triggered"] == False
    assert data["response"] == "Hola, estoy bien. ¿En qué puedo ayudarte?"
    assert data["session_id"] == "session_123"
    assert data["run_id"] == "run_123"

@patch('app.services.chat.ChatService.apply_input_filters', new_callable=AsyncMock)
def test_chat_endpoint_guardrail_triggered(mock_apply_filters):
    """
    Test para un caso donde el guardrail de entrada debería activarse.
    Tests API functionality with mocked guardrails blocking response.
    """
    # Mock guardrails response (dangerous message blocked)
    mock_apply_filters.return_value = (
        "danger", 
        "toxicity_filter: Mensaje contiene lenguaje ofensivo",
        "No puedo conversar en ese tono. ¿En qué puedo ayudarte de manera respetuosa?"
    )
    
    payload = {"query": "Eres un tonto"}
    response = client.post("/api/chatbots/banking_safe/chat?use_guardrails=true", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "response" in data
    assert "filter_triggered" in data
    assert data["filter_triggered"] == True
    # Should return the template response when blocked
    assert data["response"] == "No puedo conversar en ese tono. ¿En qué puedo ayudarte de manera respetuosa?"

@patch('app.services.chat.ChatService.handle_chat', new_callable=AsyncMock)
def test_chat_endpoint_no_guardrails(mock_handle_chat):
    """
    Test para el endpoint de chat con los guardrails desactivados.
    Tests API functionality without guardrails (mocked chat response).
    """
    # Mock chat response (no guardrails, so it processes the message)
    mock_handle_chat.return_value = (
        "Entiendo tu frustración. ¿En qué puedo ayudarte con tus servicios bancarios?", 
        "session_456",
        "run_456"
    )
    
    payload = {"query": "Eres un tonto"}
    # Mismo query ofensivo, pero sin guardrails
    response = client.post("/api/chatbots/banking_unsafe/chat?use_guardrails=false", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "response" in data
    assert "session_id" in data
    # Sin guardrails, no debería haber filtro activado
    assert data["filter_triggered"] == False
    assert data["response"] == "Entiendo tu frustración. ¿En qué puedo ayudarte con tus servicios bancarios?"
    assert data["session_id"] == "session_456"

def test_chat_endpoint_chatbot_not_found():
    """
    Test para el caso donde el chatbot_id no existe.
    """
    payload = {"query": "Hola"}
    response = client.post("/api/chatbots/inexistente/chat?use_guardrails=true", json=payload)
    
    assert response.status_code == 404
    assert "detail" in response.json()
    # Corregido: el mensaje de error ahora es más específico y profesional
    expected_detail = "Chatbot configuration not found for id 'inexistente'."
    assert response.json()["detail"] == expected_detail

# --- Tests para los endpoints de lista de chatbots ---

def test_list_chatbots_endpoint():
    """Test para el endpoint GET /api/chatbots"""
    response = client.get("/api/chatbots")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "chatbots" in data
    assert isinstance(data["chatbots"], list)
    assert len(data["chatbots"]) >= 2  # banking_safe and banking_unsafe
    
    # Check structure of chatbot objects
    for chatbot in data["chatbots"]:
        assert "id" in chatbot
        assert "name" in chatbot
        assert "description" in chatbot
        assert "has_guardrails" in chatbot
        assert isinstance(chatbot["has_guardrails"], bool)

# --- Tests para endpoints de LangSmith ---

@patch('app.routers.api.LangSmithClient')
def test_langsmith_evaluate_trace_endpoint(mock_langsmith_class):
    """Test para el endpoint POST /api/langsmith/evaluate_trace/{run_id}"""
    mock_langsmith = mock_langsmith_class.return_value
    mock_langsmith.evaluate_single_trace = AsyncMock(return_value={
        "toxicity": {
            "decision": 0,
            "score": 0.1,
            "evaluation": "No toxic content detected",
            "evaluator": "toxicity"
        }
    })
    mock_langsmith.evaluator_names = ["toxicity", "hallucination"]
    
    response = client.post("/api/langsmith/evaluate_trace/run_123")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "run_id" in data
    assert "evaluation_results" in data
    assert "evaluators_used" in data
    assert data["run_id"] == "run_123"
    mock_langsmith.evaluate_single_trace.assert_called_once_with("run_123", None)

@patch('app.routers.api.LangSmithClient')
def test_langsmith_evaluators_endpoint(mock_langsmith_class):
    """Test para el endpoint GET /api/langsmith/evaluators"""
    mock_langsmith = mock_langsmith_class.return_value
    mock_langsmith.get_available_evaluators.return_value = {
        "toxicity": {"name": "toxicity", "type": "criteria"},
        "hallucination": {"name": "hallucination", "type": "criteria"}
    }
    
    response = client.get("/api/langsmith/evaluators")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "evaluators" in data
    assert "count" in data
    assert isinstance(data["evaluators"], dict)
    assert data["count"] == 2

@patch('app.routers.api.LangSmithClient')
def test_langsmith_evaluate_response_with_run_id(mock_langsmith_class):
    """Test para el endpoint POST /api/langsmith/evaluate_response con run_id"""
    mock_langsmith = mock_langsmith_class.return_value
    mock_langsmith.evaluate_and_add_feedback = AsyncMock(return_value={
        "toxicity": {
            "decision": 0,
            "score": 0.1,
            "evaluation": "No toxic content detected",
            "evaluator": "toxicity"
        }
    })
    
    payload = {
        "prompt": "¿Cuál es la capital de Francia?",
        "response": "La capital de Francia es París."
    }
    
    response = client.post("/api/langsmith/evaluate_response?run_id=run_123", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "evaluation_results" in data
    assert "langsmith_feedback_added" in data
    assert "run_id" in data
    assert data["langsmith_feedback_added"] == True
    assert data["run_id"] == "run_123"

@patch('app.routers.api.LangSmithClient')
def test_langsmith_evaluate_response_without_run_id(mock_langsmith_class):
    """Test para el endpoint POST /api/langsmith/evaluate_response sin run_id"""
    mock_langsmith = mock_langsmith_class.return_value
    mock_evaluator_manager = MagicMock()
    mock_evaluator_manager.evaluate_response = AsyncMock(return_value={
        "toxicity": {
            "decision": 0,
            "score": 0.1,
            "evaluation": "No toxic content detected",
            "evaluator": "toxicity"
        }
    })
    mock_langsmith.evaluator_manager = mock_evaluator_manager
    
    payload = {
        "prompt": "¿Cuál es la capital de Francia?",
        "response": "La capital de Francia es París."
    }
    
    response = client.post("/api/langsmith/evaluate_response", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "evaluation_results" in data
    assert "langsmith_feedback_added" in data
    assert data["langsmith_feedback_added"] == False

@patch('app.routers.api.LangSmithClient')
def test_langsmith_evaluate_trace_readonly_endpoint(mock_langsmith_class):
    """Test para el endpoint GET /api/langsmith/evaluate_trace_readonly/{run_id}"""
    mock_langsmith = mock_langsmith_class.return_value
    mock_langsmith.evaluate_trace_readonly = AsyncMock(return_value={
        "toxicity": {
            "decision": 0,
            "score": 0.1,
            "evaluation": "No toxic content detected",
            "evaluator": "toxicity"
        }
    })
    mock_langsmith.evaluator_names = ["toxicity", "hallucination"]
    
    response = client.get("/api/langsmith/evaluate_trace_readonly/run_123")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "run_id" in data
    assert "evaluation_results" in data
    assert "evaluators_used" in data
    assert "feedback_added" in data
    assert data["run_id"] == "run_123"
    assert data["feedback_added"] == False
    mock_langsmith.evaluate_trace_readonly.assert_called_once_with("run_123", None)

@patch('app.routers.api.LangSmithClient')
def test_langsmith_evaluate_dataset_endpoint(mock_langsmith_class):
    """Test para el endpoint POST /api/langsmith/evaluate_dataset/{dataset_id}"""
    mock_langsmith = mock_langsmith_class.return_value
    mock_langsmith.evaluate_dataset = AsyncMock(return_value={
        "dataset_id": "dataset_123",
        "dataset_name": "Test Dataset",
        "total_examples": 2,
        "evaluations": {
            "example_1": {
                "evaluations": {
                    "toxicity": {"decision": 0, "score": 0.1, "evaluation": "Safe"}
                }
            }
        }
    })
    mock_langsmith.evaluator_names = ["toxicity", "hallucination"]
    
    payload = {
        "evaluator_names": ["toxicity"],
        "add_feedback": True
    }
    
    response = client.post("/api/langsmith/evaluate_dataset/dataset_123", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "dataset_evaluation" in data
    assert "evaluators_used" in data
    assert "feedback_added" in data
    assert data["feedback_added"] == True
    mock_langsmith.evaluate_dataset.assert_called_once_with("dataset_123", ["toxicity"], True)

@patch('app.routers.api.LangSmithClient')
def test_langsmith_human_feedback_endpoint(mock_langsmith_class):
    """Test para el endpoint POST /api/langsmith/human_feedback/{run_id}"""
    mock_langsmith = mock_langsmith_class.return_value
    mock_langsmith.record_human_feedback = AsyncMock(return_value={"status": "success", "feedback_id": "feedback_123"})
    
    payload = {
        "feedback_type": "thumbs",
        "value": "up",
        "comment": "Great response!"
    }
    
    response = client.post("/api/langsmith/human_feedback/run_123", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "run_id" in data
    assert "feedback_added" in data
    assert "feedback_type" in data
    assert "value" in data
    assert "comment" in data
    assert "result" in data
    assert data["run_id"] == "run_123"
    assert data["feedback_added"] == True
    assert data["feedback_type"] == "thumbs"
    assert data["value"] == "up"
    mock_langsmith.record_human_feedback.assert_called_once_with(
        run_id="run_123",
        feedback_type="thumbs", 
        value="up",
        comment="Great response!"
    )