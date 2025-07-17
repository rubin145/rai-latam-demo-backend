import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from .routers import api
from .services.langchain_chat import LangChainChatService


# Load environment variables
load_dotenv()

# Create FastAPI application
app = FastAPI(
    title="RAI Latam Demo API",
    description="API for demonstrating guardrails and evaluation in LLM based agents",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

@app.on_event("startup")
def startup_event():
    """Handles application startup events."""
    print("🚀 Multi-chatbot service ready")
    print("📁 Available chatbots: banking")
    print("🔄 Services created dynamically per request")

# Configure CORS
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
print(f"🌐 CORS Origins configured: {origins}")

# Temporary hardcode for debugging - replace with origins variable after testing
hardcoded_origins = [
    "http://localhost:3000",
    "https://rai-latam-demo-frontend.onrender.com",
    "https://rai-latam-demo-frontend.onrender.com/",
    "https://rai-latam-demo.onrender.com",
    "https://rai-latam-demo.onrender.com/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=hardcoded_origins,  # Using hardcoded for debugging
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
