"""
Dependency injection setup for FastAPI services
"""
import os
from functools import lru_cache
from typing import Annotated
from fastapi import Depends

from .services.evaluation_service import EvaluationService


@lru_cache()
def get_evaluation_service(project_name: str) -> EvaluationService:
    """
    Create and cache EvaluationService instances per project.
    
    Args:
        project_name: Name of the project (e.g., 'banking')
        
    Returns:
        EvaluationService instance configured for the project
    """
    config_path = f"configs/evaluators/llm_evaluators.yaml"
    
    # Validate that the configuration file exists
    if not os.path.exists(config_path):
        raise ValueError(f"Evaluator configuration not found at {config_path}")
    
    return EvaluationService(
        project_name=project_name,
        config_path=config_path
    )


def get_banking_evaluation_service() -> EvaluationService:
    """Dependency function for banking evaluation service"""
    return get_evaluation_service("banking")


# Type annotations for FastAPI dependency injection
BankingEvaluationService = Annotated[EvaluationService, Depends(get_banking_evaluation_service)]


def get_project_evaluation_service(project_name: str):
    """
    Factory function to create project-specific evaluation service dependencies
    
    Usage:
    @app.post("/chat/{project_name}")
    async def chat(
        project_name: str,
        eval_service: EvaluationService = Depends(get_project_evaluation_service(project_name))
    ):
    """
    def _get_service() -> EvaluationService:
        return get_evaluation_service(project_name)
    
    return _get_service