from typing import Dict, List, Optional, Any
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.utils.config import load_config

from src.embeddings.titan import TitanEmbeddings
from src.storage.qdrant import QdrantStorage
from src.utils.config import (
    RECURSIVE_COLLECTION_NAME,
    RECURSIVE_CHUNK_SIZE,
    RECURSIVE_CHUNK_OVERLAP
)

# Configure logging
logger = logging.getLogger(__name__)

class RecursiveChunker:
    """
    Recursive chunking implementation using pure RecursiveCharacterTextSplitter.
    """
    
    def __init__(self):
        """Initialize the recursive chunker with configuration."""
        self.chunk_size = int(load_config("RECURSIVE_CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(load_config("RECURSIVE_CHUNK_OVERLAP", "200"))
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        self.embeddings = TitanEmbeddings()
        self.storage = QdrantStorage()
    
    def chunk_documents(self, documents: List[Any]) -> List[Dict[str, Any]]:
        """
        Chunk documents using simple recursive character text splitting.
        No semantic understanding, just pure text splitting.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        try:
            all_text = []
            base_metadata = {}
            
            # Concatenate all document content with newlines
            for doc in documents:
                if isinstance(doc, Document):
                    all_text.append(doc.page_content)
                    # Keep track of common metadata
                    if not base_metadata:
                        base_metadata = doc.metadata
                else:
                    all_text.append(doc.get("text", ""))
                    if not base_metadata:
                        base_metadata = doc.get("metadata", {})
            
            # Join all text with double newlines to preserve some structure
            combined_text = "\n\n".join(text for text in all_text if text.strip())
            
            # Create chunks from the combined text
            chunks = self.text_splitter.create_documents(
                texts=[combined_text],
                metadatas=[{
                    **base_metadata,
                    "chunking_type": "recursive",
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap
                }]
            )
            
            # Convert to consistent format
            chunked_documents = []
            for chunk in chunks:
                chunked_documents.append({
                    "content": chunk.page_content,
                    "metadata": chunk.metadata
                })
            
            logger.info(f"Created {len(chunked_documents)} chunks using recursive chunking")
            return chunked_documents
            
        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            raise
    
    def store_chunks(self, chunks: List[Dict], collection_name: Optional[str] = None) -> None:
        """
        Store chunks in Qdrant.
        
        Args:
            chunks: List of chunks
            collection_name: Optional custom collection name
        """
        try:
            if collection_name is None:
                collection_name = RECURSIVE_COLLECTION_NAME
            
            # Extract texts and metadata
            texts = []
            metadatas = []
            for chunk in chunks:
                texts.append(chunk["content"])
                metadatas.append(chunk.get("metadata", {}))
            
            # Store in Qdrant
            self.storage.store_documents(
                collection_name=collection_name,
                texts=texts,
                embeddings=self.embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"Stored {len(chunks)} chunks in collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Error storing chunks: {e}")
            raise
    
    def process_and_store(self, elements: List[Dict], collection_name: Optional[str] = None) -> List[Dict]:
        """
        Process elements and store chunks in Qdrant.
        Simple recursive chunking without semantic understanding.
        
        Args:
            elements: List of elements to process
            collection_name: Optional custom collection name
            
        Returns:
            List of processed chunks
        """
        try:
            # Create chunks
            chunks = self.chunk_documents(elements)
            
            # Store chunks
            self.store_chunks(chunks, collection_name)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error in process_and_store: {e}")
            raise 