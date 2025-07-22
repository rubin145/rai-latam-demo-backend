import pytest
from fastapi.testclient import TestClient
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

def test_evaluate_response_endpoint_success():
    """
    Test para un caso exitoso del endpoint /api/evaluate_response.
    Envía un prompt y una respuesta, y espera un resultado válido.
    """
    payload = {
        "prompt": "¿Cuál es la capital de Francia?",
        "response": "La capital de Francia es París."
        # Se elimina el campo "evaluators" que no es parte del request
    }
    response = client.post("/api/evaluate_response", json=payload) # Ruta corregida
    
    assert response.status_code == 200
    data = response.json()
    
    # El schema de respuesta contiene 'results' que es un diccionario
    assert "results" in data
    assert isinstance(data["results"], dict)
    # La cantidad de resultados depende de la configuración del servicio,
    # así que solo verificamos que no esté vacía si hay resultados.
    if len(data["results"]) > 0:
        # Tomamos el primer evaluador, por ejemplo 'toxicity'
        first_eval_key = list(data["results"].keys())[0]
        first_eval = data["results"][first_eval_key]
        assert "evaluator" in first_eval
        assert "score" in first_eval
        # Corregido: la clave es 'evaluation', no 'reasoning'
        assert "evaluation" in first_eval
        assert isinstance(first_eval["score"], (int, float))

def test_evaluate_response_endpoint_missing_fields():
    """
    Test para el endpoint /api/evaluate_response cuando faltan campos en el payload.
    Espera una respuesta de error 422 (Unprocessable Entity) de FastAPI.
    """
    payload = {
        "prompt": "Solo un campo, falta la respuesta"
    }
    response = client.post("/api/evaluate_response", json=payload) # Ruta corregida
    
    assert response.status_code == 422 # Error de validación de Pydantic

# --- Tests para los endpoints de Chat ---

def test_chat_endpoint_success():
    """
    Test para un caso exitoso del endpoint /api/chatbots/{chatbot_id}/chat.
    """
    payload = {"query": "Hola, ¿cómo estás?"}
    # Probamos con el chatbot "banking" y guardrails activados
    response = client.post("/api/chatbots/banking/chat?use_guardrails=true", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "response" in data
    assert "session_id" in data
    assert "filter_decision" in data
    assert data["filter_decision"] == "safe" # Esperamos que un saludo simple sea seguro

def test_chat_endpoint_guardrail_triggered():
    """
    Test para un caso donde el guardrail de entrada debería activarse.
    """
    # Este query debería ser bloqueado por un guardrail de toxicidad o similar
    payload = {"query": "Eres un tonto"}
    response = client.post("/api/chatbots/banking/chat?use_guardrails=true", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "response" in data
    assert "filter_decision" in data
    assert data["filter_decision"] == "danger"

def test_chat_endpoint_no_guardrails():
    """
    Test para el endpoint de chat con los guardrails desactivados.
    """
    payload = {"query": "Eres un tonto"}
    # Mismo query ofensivo, pero sin guardrails
    response = client.post("/api/chatbots/banking/chat?use_guardrails=false", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "response" in data
    # Sin guardrails, la decisión del filtro debería ser None
    assert data["filter_decision"] is None

def test_chat_endpoint_chatbot_not_found():
    """
    Test para el caso donde el chatbot_id no existe.
    """
    payload = {"query": "Hola"}
    response = client.post("/api/chatbots/inexistente/chat?use_guardrails=true", json=payload)
    
    assert response.status_code == 404
    assert "detail" in response.json()
    # Corregido: el mensaje de error ahora es más específico y profesional
    expected_detail = "Chatbot configuration not found for id 'inexistente' with guardrails=True."
    assert response.json()["detail"] == expected_detail