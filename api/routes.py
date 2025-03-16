from typing import Dict, List, Optional, Any
import os
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Form, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api.schemas import (
    ChunkingRequest,
    ChunkingResponse,
    QueryRequest,
    QueryResponse
)
from src.chunking.recursive import RecursiveChunker
from src.chunking.semantic import SemanticChunker
from src.rag.query import RAGQuery
from src.utils.config import SEMANTIC_COLLECTION_NAME, UPLOAD_FOLDER, RECURSIVE_COLLECTION_NAME, load_config
from src.utils.parser import parse_pdf
from src.utils.html_parser import parse_html_file
from src.embeddings.titan import TitanEmbeddings
from src.storage.qdrant import QdrantStorage
from src.utils.upload import validate_file_type, create_upload_folder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

# Configure upload folder
UPLOAD_FOLDER = load_config("UPLOAD_FOLDER", "uploads")
create_upload_folder(UPLOAD_FOLDER)

# Initialize components
recursive_chunker = RecursiveChunker()
semantic_chunker = SemanticChunker()
embeddings = TitanEmbeddings()
storage = QdrantStorage()
rag_query = RAGQuery()

# Models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    recursive: Dict
    semantic: Dict
    comparison: str

# Routes
@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a document file (PDF or HTML).
    """
    try:
        # Validate file type
        file_extension = validate_file_type(file.filename)
        
        # Save file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return JSONResponse(
            content={
                "message": "File uploaded successfully",
                "file_path": file_path
            },
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process/recursive")
async def process_recursive_chunking(
    file_path: str = Form(...),
    collection_name: Optional[str] = Form(None)
):
    """
    Process document with recursive chunking.
    
    Args:
        file_path: Path to the file to process
        collection_name: Optional custom collection name (used as-is without prefix)
    """
    try:
        # Verify file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        # Get file extension
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()
        
        # Use collection name as-is if provided, otherwise generate one
        if collection_name is None:
            collection_name = os.path.basename(file_path).replace('.', '_')
        
        # Process document
        await process_file_recursive(file_path, file_extension, collection_name)
        
        return ChunkingResponse(
            message="Document processed successfully with recursive chunking",
            collection_name=collection_name
        )
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process/semantic")
async def process_semantic_chunking(
    file_path: str = Form(...),
    collection_name: Optional[str] = Form(None)
):
    """
    Process document with semantic chunking.
    
    Args:
        file_path: Path to the file to process
        collection_name: Optional custom collection name (used as-is without prefix)
    """
    try:
        # Verify file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        # Get file extension
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()
        
        # Use collection name as-is if provided, otherwise generate one
        if collection_name is None:
            collection_name = os.path.basename(file_path).replace('.', '_')
        
        # Process document
        await process_file_semantic(file_path, file_extension, collection_name)
        
        return ChunkingResponse(
            message="Document processed successfully with semantic chunking",
            collection_name=collection_name
        )
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query")
async def query(query: Dict[str, Any]):
    """
    Query both collections and compare results with RAG analysis.
    
    Args:
        query: Query parameters including:
            - query: The question to ask
            - recursive_collection: Optional name of recursive collection
            - semantic_collection: Optional name of semantic collection
        
    Returns:
        JSON response with query results and analysis
    """
    if "query" not in query:
        raise HTTPException(status_code=400, detail="Query parameter is required")
    
    query_text = query["query"]
    recursive_collection = query.get("recursive_collection", "recursive")
    semantic_collection = query.get("semantic_collection", "semantic")
    
    try:
        # Get documents from both collections
        recursive_docs = rag_query.search_collection(recursive_collection, query_text)
        semantic_docs = rag_query.search_collection(semantic_collection, query_text)
        
        # Generate answers using RAG
        recursive_answer = rag_query.generate_answer(query_text, recursive_docs)
        semantic_answer = rag_query.generate_answer(query_text, semantic_docs)
        
        # Compare the answers
        comparison = rag_query.compare_answers(query_text, recursive_answer, semantic_answer)
        
        # Get vector similarity comparison
        vector_comparison = rag_query._compare_results(
            recursive_docs[:5] if recursive_docs else [], 
            semantic_docs[:5] if semantic_docs else []
        )
        
        return {
            "query": query_text,
            "recursive": {
                "collection": recursive_collection,
                "answer": recursive_answer,
                "chunks": [
                    {"content": doc.page_content, "metadata": doc.metadata}
                    for doc in recursive_docs
                ] if recursive_docs else []
            },
            "semantic": {
                "collection": semantic_collection,
                "answer": semantic_answer,
                "chunks": [
                    {"content": doc.page_content, "metadata": doc.metadata}
                    for doc in semantic_docs
                ] if semantic_docs else []
            },
            "analysis": {
                "rag_comparison": comparison,
                "vector_comparison": vector_comparison
            }
        }
        
    except Exception as e:
        logger.error(f"Error querying collections: {e}")
        raise HTTPException(status_code=500, detail=f"Error querying collections: {str(e)}")

@router.get("/collections")
async def list_collections():
    """
    List all collections.
    
    Returns:
        JSON response with collections
    """
    try:
        collections = storage.list_collections()
        
        # Get collection info
        collection_info = []
        for collection in collections:
            try:
                info = storage.get_collection_info(collection)
                collection_info.append({
                    "name": collection,
                    "vector_count": info.get("vector_count", 0),
                    "chunking_type": "recursive" if collection.startswith("recursive_") else "semantic" if collection.startswith("semantic_") else "unknown"
                })
            except Exception as e:
                logger.error(f"Error getting info for collection {collection}: {e}")
                collection_info.append({
                    "name": collection,
                    "vector_count": 0,
                    "chunking_type": "recursive" if collection.startswith("recursive_") else "semantic" if collection.startswith("semantic_") else "unknown",
                    "error": str(e)
                })
        
        return {"collections": collection_info}
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing collections: {str(e)}")

@router.post("/collections/create")
async def create_collection(collection_data: Dict[str, Any]):
    """
    Create a new collection.
    
    Args:
        collection_data: Collection data
        
    Returns:
        JSON response with the collection name
    """
    if "name" not in collection_data:
        raise HTTPException(status_code=400, detail="Collection name is required")
    
    collection_name = collection_data["name"]
    
    try:
        storage.create_collection(collection_name)
        return {"message": f"Collection {collection_name} created successfully"}
    except Exception as e:
        logger.error(f"Error creating collection {collection_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating collection: {str(e)}")

@router.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """
    Delete a collection.
    
    Args:
        collection_name: Collection name
        
    Returns:
        JSON response with the collection name
    """
    try:
        storage.delete_collection(collection_name)
        return {"message": f"Collection {collection_name} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting collection {collection_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}")

@router.post("/collections/rename")
async def rename_collection(rename_data: Dict[str, Any]):
    """
    Rename a collection.
    
    Args:
        rename_data: Rename data
        
    Returns:
        JSON response with the collection name
    """
    if "old_name" not in rename_data or "new_name" not in rename_data:
        raise HTTPException(status_code=400, detail="Both old_name and new_name are required")
    
    old_name = rename_data["old_name"]
    new_name = rename_data["new_name"]
    
    try:
        # Create new collection
        storage.create_collection(new_name)
        
        # Copy data from old collection to new collection
        storage.copy_collection(old_name, new_name)
        
        # Delete old collection
        storage.delete_collection(old_name)
        
        return {"message": f"Collection {old_name} renamed to {new_name} successfully"}
    except Exception as e:
        logger.error(f"Error renaming collection {old_name} to {new_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error renaming collection: {str(e)}")

@router.get("/fix-collections")
async def fix_collections():
    """
    Fix collection names.
    
    Returns:
        JSON response with fixed collections
    """
    try:
        collections = storage.list_collections()
        fixed_collections = []
        
        for collection in collections:
            # Check if collection name has invalid characters
            if not collection.isalnum() and not all(c.isalnum() or c == '_' for c in collection):
                # Create a valid name
                valid_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in collection)
                
                # Rename collection
                try:
                    # Create new collection
                    storage.create_collection(valid_name)
                    
                    # Copy data from old collection to new collection
                    storage.copy_collection(collection, valid_name)
                    
                    # Delete old collection
                    storage.delete_collection(collection)
                    
                    fixed_collections.append({"old_name": collection, "new_name": valid_name})
                except Exception as e:
                    logger.error(f"Error fixing collection {collection}: {e}")
        
        return {"fixed_collections": fixed_collections}
    except Exception as e:
        logger.error(f"Error fixing collections: {e}")
        raise HTTPException(status_code=500, detail=f"Error fixing collections: {str(e)}")

# Background processing functions
async def process_file_recursive(file_path: str, file_extension: str, collection_name: str):
    """
    Process a file using recursive chunking.
    
    Args:
        file_path: Path to the file
        file_extension: File extension
        collection_name: Collection name
    """
    try:
        # Parse the file based on its type
        if file_extension == '.pdf':
            logger.info(f"Processing PDF file: {file_path}")
            documents = parse_pdf(file_path)
        elif file_extension in ['.html', '.htm']:
            logger.info(f"Processing HTML file: {file_path}")
            documents = parse_html_file(file_path)
        else:
            logger.error(f"Unsupported file type: {file_extension}")
            return
        
        logger.info(f"Parsed {len(documents)} documents from {file_path}")
        
        # Chunk the documents
        chunked_documents = recursive_chunker.chunk_documents(documents)
        logger.info(f"Created {len(chunked_documents)} chunks")
        
        # Extract texts and metadata from chunks
        texts = []
        metadatas = []
        for chunk in chunked_documents:
            if isinstance(chunk, dict):
                texts.append(chunk["content"])
                metadatas.append(chunk.get("metadata", {}))
            else:
                texts.append(chunk.page_content)
                metadatas.append(chunk.metadata)
        
        # Create collection and store documents
        storage.create_collection(collection_name)
        storage.store_documents(
            collection_name=collection_name,
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"Successfully processed {file_path} with recursive chunking")
    except Exception as e:
        logger.error(f"Error processing {file_path} with recursive chunking: {e}")
        raise

async def process_file_semantic(file_path: str, file_extension: str, collection_name: str):
    """
    Process a file using semantic chunking.
    
    Args:
        file_path: Path to the file
        file_extension: File extension
        collection_name: Collection name
    """
    try:
        # Parse the file based on its type
        if file_extension == '.pdf':
            logger.info(f"Processing PDF file: {file_path}")
            documents = parse_pdf(file_path)
        elif file_extension in ['.html', '.htm']:
            logger.info(f"Processing HTML file: {file_path}")
            documents = parse_html_file(file_path)
        else:
            logger.error(f"Unsupported file type: {file_extension}")
            return
        
        logger.info(f"Parsed {len(documents)} documents from {file_path}")
        
        # Chunk the documents
        chunked_documents = semantic_chunker.chunk_documents(documents)
        logger.info(f"Created {len(chunked_documents)} chunks")
        
        # Extract texts and metadata from chunks
        texts = []
        metadatas = []
        for chunk in chunked_documents:
            if isinstance(chunk, dict):
                texts.append(chunk["content"])
                metadatas.append(chunk.get("metadata", {}))
            else:
                texts.append(chunk.page_content)
                metadatas.append(chunk.metadata)
        
        # Create collection and store documents
        storage.create_collection(collection_name)
        storage.store_documents(
            collection_name=collection_name,
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"Successfully processed {file_path} with semantic chunking")
    except Exception as e:
        logger.error(f"Error processing {file_path} with semantic chunking: {e}")
        raise 