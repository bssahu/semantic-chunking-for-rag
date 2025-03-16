"""
Qdrant vector storage implementation.
"""

import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.utils.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_PREFIX,
    load_config
)

logger = logging.getLogger(__name__)

class QdrantStorage:
    """
    Qdrant vector storage implementation.
    """
    
    def __init__(self):
        """Initialize Qdrant client."""
        self.host = load_config("QDRANT_HOST", "localhost")
        self.port = int(load_config("QDRANT_PORT", "6333"))
        self.prefix = load_config("QDRANT_COLLECTION_PREFIX", "semantic_chunking_")
        
        self.client = QdrantClient(
            host=self.host,
            port=self.port
        )
        logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
    
    def create_collection(self, collection_name: str, vector_size: int = 1536) -> None:
        """
        Create a new collection.
        
        Args:
            collection_name: Name of the collection
            vector_size: Size of the vectors (default: 1536 for Titan embeddings)
        """
        try:
            # Add prefix to collection name if not already present
            if not collection_name.startswith(self.prefix):
                collection_name = f"{self.prefix}{collection_name}"
            
            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Error creating collection '{collection_name}': {e}")
            raise
    
    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection.
        
        Args:
            collection_name: Name of the collection
        """
        try:
            # Add prefix to collection name if not already present
            if not collection_name.startswith(self.prefix):
                collection_name = f"{self.prefix}{collection_name}"
            
            # Delete collection
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Error deleting collection '{collection_name}': {e}")
            raise
    
    def list_collections(self) -> List[str]:
        """
        List all collections.
        
        Returns:
            List of collection names
        """
        try:
            collections = self.client.get_collections()
            return [c.name for c in collections.collections]
            
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            raise
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary with collection information
        """
        try:
            # Add prefix to collection name if not already present
            if not collection_name.startswith(self.prefix):
                collection_name = f"{self.prefix}{collection_name}"
            
            # Get collection info
            collection = self.client.get_collection(collection_name=collection_name)
            
            return {
                "name": collection.name,
                "vector_size": collection.config.params.vectors.size,
                "vector_count": collection.vectors_count,
                "status": collection.status
            }
            
        except Exception as e:
            logger.error(f"Error getting info for collection '{collection_name}': {e}")
            raise
    
    def store_documents(
        self,
        collection_name: str,
        texts: List[str],
        embeddings: Any,
        metadatas: Optional[List[Dict]] = None
    ) -> None:
        """
        Store documents in a collection.
        
        Args:
            collection_name: Name of the collection
            texts: List of text documents
            embeddings: Embeddings model to use
            metadatas: Optional list of metadata dictionaries
        """
        try:
            # Add prefix to collection name if not already present
            if not collection_name.startswith(self.prefix):
                collection_name = f"{self.prefix}{collection_name}"
            
            # Create collection if it doesn't exist
            try:
                self.create_collection(collection_name)
            except Exception:
                # Collection might already exist
                pass
            
            # Generate embeddings
            vectors = embeddings.embed_documents(texts)
            
            # Prepare points
            points = []
            for i, (text, vector) in enumerate(zip(texts, vectors)):
                point = models.PointStruct(
                    id=i,
                    vector=vector,
                    payload={
                        "text": text,
                        **(metadatas[i] if metadatas else {})
                    }
                )
                points.append(point)
            
            # Upload points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
            
            logger.info(f"Stored {len(texts)} documents in collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Error storing documents in collection '{collection_name}': {e}")
            raise
    
    def search_documents(
        self,
        collection_name: str,
        query: str,
        embeddings: Any,
        limit: int = 5
    ) -> List[Dict]:
        """
        Search for similar documents in a collection.
        
        Args:
            collection_name: Name of the collection
            query: Query text
            embeddings: Embeddings model to use
            limit: Maximum number of results to return
            
        Returns:
            List of documents with similarity scores
        """
        try:
            # Add prefix to collection name if not already present
            if not collection_name.startswith(self.prefix):
                collection_name = f"{self.prefix}{collection_name}"
            
            # Generate query embedding
            query_vector = embeddings.embed_query(query)
            
            # Search collection
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit
            )
            
            # Format results
            documents = []
            for result in results:
                documents.append({
                    "text": result.payload["text"],
                    "metadata": {k: v for k, v in result.payload.items() if k != "text"},
                    "score": result.score
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error searching collection '{collection_name}': {e}")
            raise 