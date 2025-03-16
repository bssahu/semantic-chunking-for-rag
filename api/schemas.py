"""
API request and response schemas.
"""

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field

class ChunkingRequest(BaseModel):
    """Request model for document chunking."""
    file_path: str = Field(..., description="Path to the document file (PDF or HTML)")
    collection_name: Optional[str] = Field(None, description="Optional name for the collection")

class ChunkingResponse(BaseModel):
    """Response model for document chunking."""
    message: str
    collection_name: str

class QueryRequest(BaseModel):
    """Request model for querying documents."""
    query: str = Field(..., description="Query text")
    recursive_collection: Optional[str] = Field(None, description="Name of the recursive chunking collection")
    semantic_collection: Optional[str] = Field(None, description="Name of the semantic chunking collection")

class QueryResponse(BaseModel):
    """Response model for query results."""
    query: str
    recursive: Dict
    semantic: Dict
    comparison: str

class CollectionInfo(BaseModel):
    """Model for collection information."""
    name: str
    vector_count: int
    chunking_type: str
    error: Optional[str] = None

class CollectionsResponse(BaseModel):
    """Response model for listing collections."""
    collections: List[CollectionInfo]

class CollectionCreateRequest(BaseModel):
    """Request model for creating a collection."""
    name: str = Field(..., description="Name of the collection to create")

class CollectionRenameRequest(BaseModel):
    """Request model for renaming a collection."""
    old_name: str = Field(..., description="Current name of the collection")
    new_name: str = Field(..., description="New name for the collection")

class SourceDocument(BaseModel):
    """
    Schema for source document.
    """
    content: str = Field(..., description="Document content")
    metadata: Dict = Field(..., description="Document metadata")

class ChunkingResult(BaseModel):
    """
    Schema for chunking result.
    """
    answer: str = Field(..., description="Answer from the LLM")
    source_documents: List[SourceDocument] = Field(..., description="Source documents used for the answer") 