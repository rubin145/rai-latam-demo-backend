from typing import Dict, Any, Optional, List

def normalize_score(value: Any, scale: int = 3) -> float:
    """Normalize a score to a 0-1 range based on the given scale."""
    try:
        return float(value) / scale
    except (ValueError, TypeError):
        return 0.0