from typing import Dict, List, Optional, Any
import logging
import json
import re

import boto3
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from qdrant_client.http.exceptions import UnexpectedResponse

from src.embeddings.titan import TitanEmbeddings
from src.storage.qdrant import QdrantStorage
from src.utils.config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    CLAUDE_MODEL_ID,
    RECURSIVE_COLLECTION_NAME,
    SEMANTIC_COLLECTION_NAME,
    DEFAULT_COLLECTION_NAME
)

# Configure logging
logger = logging.getLogger(__name__)

class RAGQuery:
    """
    RAG query implementation with comparison between recursive and semantic chunking.
    """
    
    def __init__(self):
        """Initialize RAG query with components."""
        self.embeddings = TitanEmbeddings()
        self.storage = QdrantStorage()
        self._bedrock_client = None
        self._llm = None
        
        # Initialize Bedrock client and LLM
        try:
            self._bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=AWS_REGION,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY
            )
            
            self._llm = ChatBedrock(
                client=self._bedrock_client,
                model_id=CLAUDE_MODEL_ID,
                model_kwargs={
                    "temperature": 0.2,
                    "max_tokens": 1000,
                    "top_p": 0.9,
                }
            )
            logger.info("Successfully initialized Bedrock client and LLM")
        except Exception as e:
            logger.error(f"Error initializing Bedrock client and LLM: {e}")
            raise
    
    def query(self, query_text: str, recursive_collection: str, semantic_collection: str) -> Dict[str, Any]:
        """
        Query both collections and compare results.
        
        Args:
            query_text: Query text
            recursive_collection: Name of the recursive chunking collection
            semantic_collection: Name of the semantic chunking collection
            
        Returns:
            Dictionary with query results and comparison
        """
        try:
            # Verify collections exist
            collections = self.storage.list_collections()
            if recursive_collection not in collections:
                raise ValueError(f"Collection '{recursive_collection}' not found")
            if semantic_collection not in collections:
                raise ValueError(f"Collection '{semantic_collection}' not found")
            
            # Query recursive collection
            recursive_results = self.storage.search_documents(
                collection_name=recursive_collection,
                query=query_text,
                embeddings=self.embeddings,
                limit=5
            )
            
            # Query semantic collection
            semantic_results = self.storage.search_documents(
                collection_name=semantic_collection,
                query=query_text,
                embeddings=self.embeddings,
                limit=5
            )
            
            # Compare results
            comparison = self._compare_results(recursive_results, semantic_results)
            
            return {
                "query": query_text,
                "recursive": {
                    "collection": recursive_collection,
                    "results": recursive_results
                },
                "semantic": {
                    "collection": semantic_collection,
                    "results": semantic_results
                },
                "comparison": comparison
            }
            
        except Exception as e:
            logger.error(f"Error querying collections: {e}")
            raise
    
    def _compare_results(self, recursive_docs: List[Document], semantic_docs: List[Document]) -> str:
        """
        Compare results from both collections.
        
        Args:
            recursive_docs: Results from recursive chunking
            semantic_docs: Results from semantic chunking
            
        Returns:
            Comparison analysis
        """
        try:
            # Compare number of results
            rec_count = len(recursive_docs)
            sem_count = len(semantic_docs)
            
            # Generate comparison text
            comparison = []
            comparison.append(f"Found {rec_count} recursive chunks and {sem_count} semantic chunks.")
            
            # Compare content types
            rec_types = set(doc.metadata.get("type", "unknown") for doc in recursive_docs)
            sem_types = set(doc.metadata.get("type", "unknown") for doc in semantic_docs)
            
            if "table" in sem_types:
                comparison.append("Semantic chunking found relevant table data.")
            if len(sem_types) > len(rec_types):
                comparison.append("Semantic chunking provided more diverse content types.")
            
            # Compare metadata richness
            rec_metadata_keys = set()
            sem_metadata_keys = set()
            for doc in recursive_docs:
                rec_metadata_keys.update(doc.metadata.keys())
            for doc in semantic_docs:
                sem_metadata_keys.update(doc.metadata.keys())
                
            if len(sem_metadata_keys) > len(rec_metadata_keys):
                comparison.append("Semantic chunking preserved more metadata.")
                
            # Compare content structure
            rec_structured = any("section" in doc.metadata for doc in recursive_docs)
            sem_structured = any("section" in doc.metadata for doc in semantic_docs)
            
            if sem_structured and not rec_structured:
                comparison.append("Semantic chunking better preserved document structure.")
            
            return "\n".join(comparison)
            
        except Exception as e:
            logger.error(f"Error comparing results: {e}")
            return "Error comparing results"
    
    def is_table_query(self, query: str) -> bool:
        """
        Determine if a query is likely about tabular data.
        
        Args:
            query: Query text
            
        Returns:
            True if the query is likely about tables
        """
        table_keywords = [
            "table", "row", "column", "cell", "data", 
            "value", "statistic", "average", "mean", "sum",
            "total", "maximum", "minimum", "count", "percentage",
            "compare", "comparison", "trend", "growth", "decline",
            "increase", "decrease", "ratio", "proportion", "distribution"
        ]
        
        query_lower = query.lower()
        
        # Check for table-related keywords
        for keyword in table_keywords:
            if re.search(r'\b' + keyword + r'\b', query_lower):
                return True
        
        return False
    
    def search_collection(self, collection_name: str, query: str, k: int = 5) -> List[Document]:
        """
        Search a collection for relevant documents.
        
        Args:
            collection_name: Name of the collection
            query: Query text
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        try:
            # Check if this is a table-related query
            is_table_query = self.is_table_query(query)
            
            # Search in Qdrant
            results = self.storage.search_documents(
                collection_name=collection_name,
                query=query,
                embeddings=self.embeddings,
                limit=k + (3 if is_table_query else 0)  # Get extra results for table queries
            )
            
            # Convert results to Documents
            documents = []
            for result in results:
                documents.append(Document(
                    page_content=result["text"],  # Changed from content to text
                    metadata=result["metadata"]
                ))
            
            # For table queries in semantic collection, prioritize structured table formats
            if is_table_query and collection_name == SEMANTIC_COLLECTION_NAME:
                # Group documents by table_id
                table_docs = {}
                text_docs = []
                
                for doc in documents:
                    if doc.metadata.get("type") == "table":
                        table_id = doc.metadata.get("table_id", "unknown")
                        if table_id not in table_docs:
                            table_docs[table_id] = []
                        table_docs[table_id].append(doc)
                    else:
                        text_docs.append(doc)
                
                # Prioritize documents based on format for each table
                prioritized_docs = []
                
                # First add one document of each relevant table
                for table_id, docs in table_docs.items():
                    # Sort by purpose: query > analysis > overview > display
                    sorted_docs = sorted(docs, key=lambda d: {
                        "query": 0, 
                        "analysis": 1, 
                        "overview": 2, 
                        "display": 3
                    }.get(d.metadata.get("table_purpose", ""), 4))
                    
                    # Add the most relevant document for this table
                    if sorted_docs:
                        prioritized_docs.append(sorted_docs[0])
                        
                        # If this table has JSON data, also include it
                        json_docs = [d for d in sorted_docs if d.metadata.get("table_format") == "json"]
                        if json_docs and json_docs[0] != sorted_docs[0]:
                            prioritized_docs.append(json_docs[0])
                
                # Then add text documents
                prioritized_docs.extend(text_docs)
                
                # Limit to k documents
                documents = prioritized_docs[:k]
            
            return documents
            
        except Exception as e:
            logger.error(f"Error searching collection '{collection_name}': {str(e)}")
            return []
    
    def format_document_for_context(self, doc: Document) -> str:
        """
        Format a document for inclusion in the context.
        
        Args:
            doc: Document to format
            
        Returns:
            Formatted document text
        """
        # Check if it's a table document
        if doc.metadata.get("type") == "table":
            table_format = doc.metadata.get("table_format", "")
            table_purpose = doc.metadata.get("table_purpose", "")
            table_id = doc.metadata.get("table_id", "unknown")
            
            # For JSON data, format it specially
            if table_format == "json" and "json_data" in doc.metadata:
                try:
                    # Parse the JSON data
                    table_data = json.loads(doc.metadata["json_data"])
                    
                    # Format as a more readable table for the LLM
                    formatted_text = f"TABLE DATA (ID: {table_id}, FORMAT: JSON):\n"
                    
                    # If it's a list of records
                    if isinstance(table_data, list) and len(table_data) > 0:
                        # Get headers from the first record
                        headers = list(table_data[0].keys())
                        formatted_text += " | ".join(headers) + "\n"
                        formatted_text += "-" * len(formatted_text) + "\n"
                        
                        # Add rows
                        for record in table_data:
                            row = [str(record.get(h, "")) for h in headers]
                            formatted_text += " | ".join(row) + "\n"
                    
                    # Add the JSON for completeness
                    formatted_text += f"\nJSON Representation:\n{doc.metadata['json_data']}"
                    
                    return formatted_text
                except Exception as e:
                    logger.warning(f"Error formatting JSON table data: {str(e)}")
            
            # Add table metadata to the content
            prefix = f"TABLE (ID: {table_id}, FORMAT: {table_format}, PURPOSE: {table_purpose}):\n"
            return prefix + doc.page_content
        
        # For regular text documents
        if doc.metadata.get("section"):
            return f"SECTION: {doc.metadata['section']}\n\n{doc.page_content}"
        
        return doc.page_content
    
    def generate_answer(self, query: str, documents: List[Document]) -> str:
        """
        Generate an answer based on the query and documents.
        
        Args:
            query: Query text
            documents: List of relevant documents
            
        Returns:
            Generated answer
        """
        # If no documents found, return a helpful message
        if not documents:
            return (
                "I don't have enough information to answer this question. "
                "It appears that no documents have been processed yet, or the collections don't exist. "
                "Please upload and process a document first using the /upload and /process endpoints."
            )
            
        # Check if this is a table-related query
        is_table_query = self.is_table_query(query)
        
        # Create appropriate prompt based on query type
        if is_table_query:
            prompt = ChatPromptTemplate.from_template("""
            <context>
            {context}
            </context>
            
            Human: Based on the context provided, please answer the following question about tabular data: {question}
            
            When analyzing tables:
            1. Extract relevant data points from the tables
            2. Perform calculations if needed (sums, averages, comparisons, etc.)
            3. Present the data in a clear, structured format
            4. If appropriate, create a markdown table in your response
            5. Explain what the data means and its significance
            
            Assistant: 
            """)
        else:
            prompt = ChatPromptTemplate.from_template("""
            <context>
            {context}
            </context>
            
            Human: Based on the context provided, please answer the following question thoroughly and accurately: {question}
            If the context contains tables, please analyze the table data and include relevant information in your answer.
            
            Assistant: 
            """)
        
        # Format documents for context
        formatted_docs = [self.format_document_for_context(doc) for doc in documents]
        
        # Join document contents with clear separators
        context = "\n\n---\n\n".join(formatted_docs)
        
        try:
            # Generate answer
            response = self._llm.invoke(
                prompt.format(context=context, question=query)
            )
            
            # Extract the content from the response
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Error generating answer: {str(e)}"
    
    def compare_answers(self, query: str, answer1: str, answer2: str) -> str:
        """
        Compare two answers and explain which one is better.
        
        Args:
            query: Query text
            answer1: First answer
            answer2: Second answer
            
        Returns:
            Comparison explanation
        """
        # If either answer indicates no documents, return a helpful message
        if "I don't have enough information" in answer1 or "I don't have enough information" in answer2:
            return (
                "One or both chunking methods didn't have enough information to provide a complete answer. "
                "Please ensure that documents have been processed with both chunking methods before comparing results."
            )
            
        # Check if this is a table-related query
        is_table_query = self.is_table_query(query)
        
        # Create appropriate prompt based on query type
        if is_table_query:
            prompt = ChatPromptTemplate.from_template("""
            I have two different answers to the same question about tabular data. Please compare them and explain which one is better and why.
            
            Question: {question}
            
            Answer 1 (Recursive Chunking): {answer1}
            
            Answer 2 (Semantic Chunking with Structured Tables): {answer2}
            
            Compare these answers in terms of:
            1. Accuracy of data extraction and calculations
            2. Completeness of table information
            3. Clarity of data presentation
            4. Insights derived from the tabular data
            5. Overall usefulness for understanding the tables
            
            Which answer is better overall and why? Be specific about how the structured representation of tables affected the quality of the answers.
            """)
        else:
            prompt = ChatPromptTemplate.from_template("""
            I have two different answers to the same question. Please compare them and explain which one is better and why.
            
            Question: {question}
            
            Answer 1 (Recursive Chunking): {answer1}
            
            Answer 2 (Semantic Chunking with Structured Tables): {answer2}
            
            Compare these answers in terms of:
            1. Accuracy
            2. Completeness
            3. Relevance
            4. Coherence
            5. Handling of structured data (if applicable)
            
            Which answer is better overall and why?
            """)
        
        try:
            # Generate comparison
            response = self._llm.invoke(
                prompt.format(
                    question=query,
                    answer1=answer1,
                    answer2=answer2
                )
            )
            
            # Extract the content from the response
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"Error generating comparison: {str(e)}")
            return f"Error generating comparison: {str(e)}"
    
    def _ensure_collections_exist(self):
        """
        Check if the required collections exist and create them if they don't.
        """
        try:
            # Get list of existing collections
            collections = self.storage.list_collections()
            collection_names = [c.name for c in collections]
            logger.info(f"Found existing collections: {collection_names}")
            
            # Check recursive collection
            if RECURSIVE_COLLECTION_NAME not in collection_names:
                # Check for alternative names that might exist
                alternative_names = ["recursive_financial_report", "recursive"]
                found = False
                for alt_name in alternative_names:
                    if alt_name in collection_names and alt_name != RECURSIVE_COLLECTION_NAME:
                        logger.warning(f"Found alternative collection '{alt_name}' instead of '{RECURSIVE_COLLECTION_NAME}'")
                        found = True
                        break
                
                if not found:
                    logger.warning(f"Collection '{RECURSIVE_COLLECTION_NAME}' does not exist. It will be created when documents are processed.")
            
            # Check semantic collection
            if SEMANTIC_COLLECTION_NAME not in collection_names:
                # Check for alternative names that might exist
                alternative_names = ["semantic_financial_report", "semantic"]
                found = False
                for alt_name in alternative_names:
                    if alt_name in collection_names and alt_name != SEMANTIC_COLLECTION_NAME:
                        logger.warning(f"Found alternative collection '{alt_name}' instead of '{SEMANTIC_COLLECTION_NAME}'")
                        found = True
                        break
                
                if not found:
                    logger.warning(f"Collection '{SEMANTIC_COLLECTION_NAME}' does not exist. It will be created when documents are processed.")
                
        except Exception as e:
            logger.error(f"Error checking collections: {str(e)}")
    
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
    
    @property
    def llm(self):
        """
        Lazy-load the LLM to avoid pickling issues.
        """
        if self._llm is None:
            self._llm = ChatBedrock(
                client=self.bedrock_client,
                model_id=CLAUDE_MODEL_ID,
                model_kwargs={
                    "temperature": 0.2,
                    "max_tokens": 1000,
                    "top_p": 0.9,
                }
            )
        return self._llm 