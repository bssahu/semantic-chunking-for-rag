# Semantic Chunking Architecture

## Overview

The Semantic Chunking application is designed to demonstrate the advantages of semantic document chunking over traditional recursive chunking for Retrieval-Augmented Generation (RAG) applications. The system processes PDF documents, extracts structured elements, and uses two different chunking strategies to prepare the content for retrieval.

## Components

### 1. Document Processing

- **PDF Parser**: Uses Unstructured to extract structured elements from PDF documents, including tables, text, and lists.
- **Fallback Parser**: Uses PyPDF for basic text extraction when Unstructured is not available or fails.

### 2. Chunking Strategies

- **Recursive Chunking**: Traditional approach that splits text based on character count and separators.
- **Semantic Chunking**: Advanced approach that preserves document structure and enhances table handling.
  - **Table Processing**: Converts tables to structured JSON format, extracts statistics, and creates specialized chunks.
  - **Section Preservation**: Maintains document sections and their relationships.

### 3. Vector Storage

- **Qdrant**: Vector database for storing and retrieving document chunks.
- **Collections**: Separate collections for recursive and semantic chunks to enable comparison.

### 4. Embeddings and LLM

- **Amazon Titan**: Used for generating embeddings for document chunks.
- **Amazon Claude**: Used for generating answers and comparing results.

### 5. API Layer

- **FastAPI**: Provides REST API endpoints for document uploading, processing, and querying.
- **Background Tasks**: Handles document processing asynchronously.

## Data Flow

1. **Document Upload**: PDF documents are uploaded via the API.
2. **Document Processing**: Documents are parsed into structured elements.
3. **Chunking**: Elements are processed using both recursive and semantic chunking strategies.
4. **Embedding**: Chunks are embedded using Amazon Titan.
5. **Storage**: Embeddings and metadata are stored in Qdrant collections.
6. **Querying**: User queries are processed against both collections.
7. **Answer Generation**: Claude generates answers based on retrieved chunks.
8. **Comparison**: Answers from both strategies are compared to demonstrate the advantages of semantic chunking.

## Semantic Chunking Advantages

The semantic chunking strategy offers several advantages:

1. **Structured Table Handling**: Tables are converted to JSON format, preserving their structure and enabling more accurate retrieval and analysis.
2. **Statistical Analysis**: Numeric columns in tables are analyzed to extract statistics, enhancing the system's ability to answer quantitative questions.
3. **Document Structure Preservation**: Sections and their relationships are maintained, improving context understanding.
4. **Specialized Chunks**: Different aspects of tables (overview, analysis, query) are represented as separate chunks, optimizing retrieval for different query types.

## Deployment Architecture

The application can be deployed as a standalone service or as part of a larger system. The main components are:

- **Web Server**: Hosts the FastAPI application.
- **Qdrant**: Runs as a Docker container for vector storage.
- **AWS Bedrock**: Provides embeddings and LLM capabilities.

For production deployments, consider adding:
- Load balancing
- Authentication and authorization
- Monitoring and logging
- Scaling for the web server and Qdrant 