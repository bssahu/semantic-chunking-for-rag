"""
Utilities for handling file uploads.
"""

import os
from typing import List
from fastapi import HTTPException

def validate_file_type(filename: str) -> str:
    """
    Validate file type and return the extension.
    
    Args:
        filename: Name of the file
        
    Returns:
        File extension
        
    Raises:
        HTTPException: If file type is not supported
    """
    # Get file extension
    file_extension = os.path.splitext(filename)[1].lower()
    
    # Check if file type is supported
    if file_extension not in ['.pdf', '.html', '.htm']:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only PDF and HTML files are allowed."
        )
    
    return file_extension

def get_supported_extensions() -> List[str]:
    """Get list of supported file extensions."""
    return ['.pdf', '.html', '.htm']

def create_upload_folder(folder_path: str) -> None:
    """
    Create upload folder if it doesn't exist.
    
    Args:
        folder_path: Path to the upload folder
    """
    os.makedirs(folder_path, exist_ok=True) 