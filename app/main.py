import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from .routers import api
from .services.harm_evaluator_service import HarmEvaluatorService
from .services.air_chat import AIRChatService
from .services.groq_chat import GroqChatService

# Try to import AI Refinery SDK
try:
    from air import login, DistillerClient
    AI_REFINERY_AVAILABLE = True
except ImportError:
    AI_REFINERY_AVAILABLE = False


# Load environment variables
load_dotenv()

# Create FastAPI application
app = FastAPI(
    title="Harm Evaluator API",
    description="API for evaluating potentially harmful content using AI Refinery SDK",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

@app.on_event("startup")
def startup_event():
    """Handles application startup events."""
    # Initialize AI Refinery client and projects
    if not AI_REFINERY_AVAILABLE:
        print("⚠️ AI Refinery SDK not available. The application will run in mock mode.")
        app.state.distiller_client = None
    else:
        try:
            print("Initializing AI Refinery client...")
            login(
                account=str(os.getenv("AIR_ACCOUNT")),
                api_key=str(os.getenv("AIR_API_KEY")),
            )
            client = DistillerClient()
            app.state.distiller_client = client
            print("Applying AI Refinery project configurations...")
            client.create_project(config_path="config.yaml", project=os.getenv("PROJECT_NAME", "harm_evaluator_ui"))
            client.create_project(config_path="air_chat_config.yaml", project="chat_project")
            client.create_project(config_path="air_chat_rai_config.yaml", project="chat_guardrails_project")
            print("AI Refinery client and projects initialized successfully.")
        except Exception as e:
            print(f"❌ Failed to initialize AI Refinery client: {e}")
            print("🔥 The application will run in full mock mode.")
            app.state.distiller_client = None

    # Choose chat backend services based on ENGINE (AIR or GROQ)
    engine = os.getenv("ENGINE", "AIR").upper()
    if engine == "GROQ":
        print("🔀 Using GroqChatService (base)")
        app.state.base_chat_service = GroqChatService(os.getenv("GROQ_API_KEY"),"groq_chat_config.yaml")
        print("🔀 Using GroqChatService (guardrails)")
        app.state.guardrails_chat_service = GroqChatService(os.getenv("GROQ_API_KEY"),"groq_chat_rai_config.yaml")
    else:
        print("🔀 Using AIRChatService (base)")
        app.state.base_chat_service = AIRChatService(
            app.state.distiller_client,
            os.getenv("CHAT_PROJECT", "chat_project"),
        )
        print("🔀 Using AIRChatService (guardrails)")
        app.state.guardrails_chat_service = AIRChatService(
            app.state.distiller_client,
            os.getenv("CHAT_GUARDRAILS_PROJECT", "chat_guardrails_project"),
        )

# Configure CORS
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin", "Access-Control-Request-Method", "Access-Control-Request-Headers"],
    max_age=600,
)

# Include routers
app.include_router(api.router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Harm Evaluator API",
        "version": "1.0.0",
        "docs": "/api/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", 8000))
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
