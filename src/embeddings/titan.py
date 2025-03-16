from typing import List
import json
import logging

import boto3
from langchain_aws import BedrockEmbeddings
from langchain_core.embeddings import Embeddings

from src.utils.config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    TITAN_EMBEDDING_MODEL_ID
)

# Configure logging
logger = logging.getLogger(__name__)

class TitanEmbeddings(Embeddings):
    """
    Embeddings implementation using Amazon Titan.
    """
    
    def __init__(self):
        """
        Initialize the Titan embeddings.
        """
        self._bedrock_client = None
    
    @property
    def bedrock_client(self):
        """
        Lazy-load the Bedrock client to avoid pickling issues.
        """
        if self._bedrock_client is None:
            self._bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=AWS_REGION,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY
            )
        return self._bedrock_client
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of document texts
            
        Returns:
            List of embeddings
        """
        embeddings = []
        
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query.
        
        Args:
            text: Query text
            
        Returns:
            Embedding
        """
        # Prepare request
        body = {
            "inputText": text
        }
        
        # Call Bedrock
        response = self.bedrock_client.invoke_model(
            modelId=TITAN_EMBEDDING_MODEL_ID,
            body=json.dumps(body)
        )
        
        # Parse response
        response_body = json.loads(response["body"].read())
        embedding = response_body["embedding"]
        
        return embedding
    
    def get_embeddings_model(self) -> Embeddings:
        """
        Get the underlying LangChain embeddings model.
        
        Returns:
            LangChain embeddings model
        """
        return self.embeddings_model 