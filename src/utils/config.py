"""
Configuration utilities for the application.
"""

import os
from dotenv import load_dotenv
from typing import Any, Optional, Callable, TypeVar, Union

# Load environment variables from .env file
load_dotenv()

T = TypeVar('T')

# AWS Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Bedrock model IDs
TITAN_MODEL_ID = os.getenv("TITAN_MODEL_ID", "amazon.titan-embed-text-v1")
TITAN_EMBEDDING_MODEL_ID = TITAN_MODEL_ID
CLAUDE_MODEL_ID = os.getenv("CLAUDE_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")

# Qdrant Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION_PREFIX = os.getenv("QDRANT_COLLECTION_PREFIX", "")

# Collection names with prefix
DEFAULT_COLLECTION_NAME = f"{QDRANT_COLLECTION_PREFIX}financial_report"
RECURSIVE_COLLECTION_NAME = f"{QDRANT_COLLECTION_PREFIX}recursive"
SEMANTIC_COLLECTION_NAME = f"{QDRANT_COLLECTION_PREFIX}semantic"

# Chunking Configuration
RECURSIVE_CHUNK_SIZE = int(os.getenv("RECURSIVE_CHUNK_SIZE", "1000"))
RECURSIVE_CHUNK_OVERLAP = int(os.getenv("RECURSIVE_CHUNK_OVERLAP", "200"))
SEMANTIC_CHUNK_SIZE = int(os.getenv("SEMANTIC_CHUNK_SIZE", "1000"))
SEMANTIC_CHUNK_OVERLAP = int(os.getenv("SEMANTIC_CHUNK_OVERLAP", "200"))

# API Configuration
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", "16777216"))  # 16MB

# Vector dimensions
EMBEDDING_DIMENSION = 1536

def load_config(key: str, default: T, type_converter: Optional[Callable[[str], T]] = None) -> T:
    """
    Load configuration value from environment variable with type conversion.
    
    Args:
        key: Environment variable key
        default: Default value if key is not found
        type_converter: Optional function to convert string value to desired type
        
    Returns:
        Configuration value of type T
    """
    value = os.getenv(key)
    
    if value is None:
        return default
    
    if type_converter is not None:
        try:
            return type_converter(value)
        except (ValueError, TypeError) as e:
            print(f"Warning: Failed to convert {key}={value} using {type_converter.__name__}: {e}")
            return default
    
    return value  # type: ignore 