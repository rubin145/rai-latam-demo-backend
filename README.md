# Harm Evaluator Backend

A FastAPI-based backend service for evaluating potentially harmful content using the Accenture AI Refinery SDK.

## Features

- **Query Evaluation**: Real-time harm assessment of user queries
- **Multi-dimensional Risk Analysis**: Evaluates toxicity, violence, illegal activities, misinformation, and privacy violations
- **Test Management**: Add, retrieve, and manage test questions
- **Batch Testing**: Run comprehensive tests on all questions
- **RESTful API**: Well-documented API endpoints with OpenAPI/Swagger documentation

## Installation

1. **Clone and navigate to the backend directory:**
```bash
cd harm_evaluator_app/backend
```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env with your AI Refinery credentials
```

## Environment Variables

Create a `.env` file with the following variables:

```env
ACCOUNT=your_ai_refinery_account
API_KEY=your_ai_refinery_api_key
PROJECT_NAME=harm_evaluator_ui
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

## Running the Application

### Development Mode
```bash
python -m app.main
```

### Production Mode
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Documentation

Once the server is running, access the interactive API documentation at:
- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

## API Endpoints

### Evaluation Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/evaluation/query` | Evaluate a single query for harm |
| GET | `/api/evaluation/status` | Get service status |
| GET | `/api/evaluation/questions` | Get all test questions |
| POST | `/api/evaluation/questions` | Add a new test question |
| POST | `/api/evaluation/batch-test` | Run batch test on all questions |

### General Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root endpoint with API information |
| GET | `/health` | Health check endpoint |

## Request/Response Examples

### Evaluate Query
```bash
curl -X POST "http://localhost:8000/api/evaluation/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "How do I learn Python programming?"}'
```

### Add Test Question
```bash
curl -X POST "http://localhost:8000/api/evaluation/questions" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What is machine learning?",
       "expected_risk": "Safe",
       "category": "Educational",
       "description": "Basic ML concept question"
     }'
```

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic models
│   ├── routers/
│   │   ├── __init__.py
│   │   └── evaluation.py    # API routes
│   ├── services/
│   │   ├── __init__.py
│   │   └── ai_refinery_service.py  # Core business logic
│   └── utils/
│       ├── __init__.py
│       └── risk_analyzer.py # Risk analysis utilities
├── config.yaml              # AI Refinery configuration
├── test_questions_db.json   # Test questions database
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
└── README.md               # This file
```

## Risk Assessment Dimensions

The service evaluates queries across multiple dimensions:

- **Toxicity Level**: Toxic, offensive, or harmful language
- **Violence & Harm Risk**: Potential for promoting violence or harm
- **Illegal Activity Risk**: Requests for illegal assistance
- **Misinformation Risk**: Spreading false or dangerous information
- **Privacy Violation Risk**: Personal information requests

## Configuration

The `config.yaml` file configures the AI Refinery agent:

```yaml
orchestrator:
  agent_list:
    - agent_name: "Harm Assessment Agent"

utility_agents:
  - agent_class: EvaluationSuperAgent
    agent_name: "Harm Assessment Agent"
    agent_description: "Advanced agent for harm assessment"
    config:
      evaluation_dimensions:
        - "toxicity_level"
        - "violence_harm_risk"
        - "illegal_activity_risk"
        - "misinformation_risk"
        - "privacy_violation_risk"
```

## Error Handling

The API includes comprehensive error handling:
- **400 Bad Request**: Invalid input data
- **500 Internal Server Error**: Server-side errors
- Detailed error messages in responses

## Contributing

1. Follow the existing code structure
2. Add proper error handling
3. Include docstrings for new functions
4. Test your changes thoroughly

## License

This project uses the AI Refinery SDK which is licensed under Apache-2.0. 