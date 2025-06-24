 # RAI LATAM Demo Backend

 A FastAPI-based backend service for the RAI LATAM demonstration. This service provides:

 - **Harm Evaluation**: Assess potential harm in user-provided queries across multiple risk dimensions.
 - **Chat Interface**: Conversational API end points, with optional guardrails for safe interactions.
 - **Response Evaluation**: Evaluate model-generated responses (e.g., for toxicity, misinformation).
 - **Test Management**: CRUD operations for test questions and batch-testing capabilities.

 The backend leverages the Accenture AI Refinery SDK (AIR) and Groq for AI-powered assessments and guardrails.

 ---

 ## Table of Contents

 - [Prerequisites](#prerequisites)
 - [Installation](#installation)
 - [Configuration](#configuration)
 - [Running the Application](#running-the-application)
 - [API Documentation](#api-documentation)
 - [API Endpoints](#api-endpoints)
 - [Project Structure](#project-structure)
 - [Configuration Files](#configuration-files)
 - [Contributing](#contributing)
 - [License](#license)

 ---

 ## Prerequisites

 - Python 3.12 or higher
 - [Poetry](https://python-poetry.org/)
 - (Optional) Docker & Docker Compose

 ## Installation

 1. **Clone the repository**

    ```bash
    git clone <repository-url>
    cd rai-latam-demo-backend
    ```

 2. **Install dependencies**

    ```bash
    poetry install
    ```

 ## Configuration

 Copy and edit a `.env` file in the project root with the following environment variables:

 ```env
 AIR_ACCOUNT=your_ai_refinery_account
 AIR_API_KEY=your_ai_refinery_api_key
 GROQ_API_KEY=your_groq_api_key            # Required for GROQ-based chat or evaluation
 PROJECT_NAME=harm_evaluator_ui            # AIR project for harm evaluation
 CHAT_PROJECT=chat_project                 # AIR project for standard chat
 CHAT_GUARDRAILS_PROJECT=chat_guardrails_project  # AIR project for chat guardrails
 ENGINE=AIR                                # or GROQ to switch chat backend
 CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
 BACKEND_HOST=0.0.0.0
 BACKEND_PORT=8000
 ``` 

 > **Note:** There is no `.env.example` file—create `.env` manually based on the snippet above.

 ## Running the Application

 ### Development

 ```bash
 poetry run uvicorn app.main:app --reload
 ```

 ### Production

 ```bash
 uvicorn app.main:app --host ${BACKEND_HOST:-0.0.0.0} --port ${BACKEND_PORT:-8000}
 ```

 ### Docker

 ```bash
 docker build -t rai-latam-demo-backend .
 docker run -d \
   -p 8000:8000 \
   --env-file .env \
   rai-latam-demo-backend
 ```

 ## API Documentation

 Once the server is running, API docs are available at:

 - Swagger UI: `http://localhost:8000/api/docs`
 - ReDoc:       `http://localhost:8000/api/redoc`

 ## API Endpoints

 ### Harm Evaluation

 | Method | Path                         | Description                             |
 | ------ | -----------------------------| --------------------------------------- |
 | POST   | `/api/evaluation/query`      | Evaluate a single query for harm        |
 | GET    | `/api/evaluation/status`     | Get service status                      |
 | GET    | `/api/evaluation/questions`  | List all test questions                 |
 | POST   | `/api/evaluation/questions`  | Add a new test question                 |
 | POST   | `/api/evaluation/batch-test` | Run batch tests on all saved questions  |

 ### Chat

 | Method | Path                  | Description                             |
 | ------ | ----------------------| ----------------------------------------|
 | POST   | `/api/chat`           | Standard chat without guardrails        |
 | POST   | `/api/chat-guardrails`| Chat with guardrail filters             |

 ### Response Evaluation

 | Method | Path                    | Description                                         |
 | ------ | ------------------------| ----------------------------------------------------|
 | POST   | `/api/evaluate_response`| Evaluate a model response across configured dimensions |

 ### General

 | Method | Path      | Description              |
 | ------ | ----------| -------------------------|
 | GET    | `/`       | Root endpoint info       |
 | GET    | `/health` | Health check             |

 ## Project Structure

 ```
 .
 ├── Dockerfile
 ├── config.yaml
 ├── air_chat_config.yaml
 ├── air_chat_rai_config.yaml
 ├── groq_chat_config.yaml
 ├── groq_chat_rai_config.yaml
 ├── groq_response_evaluator_config.yaml
 ├── test_questions_db.json
 ├── pyproject.toml
 ├── poetry.lock
 ├── .env              # (not committed)
 ├── app/
 │   ├── main.py
 │   ├── routers/
 │   ├── services/
 │   ├── utils/
 │   └── models/
 └── README.md
 ```

 ## Configuration Files

 - **config.yaml**                  — Main AIR agent configuration
 - **air_chat_config.yaml**         — AIR chat settings (base)
 - **air_chat_rai_config.yaml**     — AIR chat guardrails settings
 - **groq_chat_config.yaml**        — GROQ chat settings (base)
 - **groq_chat_rai_config.yaml**    — GROQ chat guardrails settings
 - **groq_response_evaluator_config.yaml** — GROQ response evaluation config

 ## Contributing

 Contributions welcome! Please open issues or PRs, follow existing style, include tests and docstrings.

 ## License

 This project uses the Accenture AI Refinery SDK (Apache-2.0).