import yaml

def load_yaml(path: str) -> dict:
    """Load a YAML configuration file and return as dict, or empty dict on error."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"⚠️ Failed to load config '{path}': {e}")
        return {}