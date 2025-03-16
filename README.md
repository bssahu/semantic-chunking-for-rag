# Semantic Document Chunking and RAG

A powerful FastAPI application for semantic document chunking and Retrieval-Augmented Generation (RAG) with enhanced table and HTML handling.

## Features

- **Advanced Document Processing**: 
  - Extract structured elements from PDF and HTML documents
  - Process tables, text, lists, and HTML elements
  - Preserve document structure and semantic relationships
- **Dual Chunking Strategies**:
  - **Recursive Chunking**: Traditional text splitting based on character count
  - **Semantic Chunking**: Intelligent chunking that preserves document structure and enhances table handling
- **Enhanced Table Processing**:
  - Converts tables to structured JSON format
  - Creates specialized chunks for different aspects of tables (overview, analysis, query)
  - Preserves table structure for better querying
  - Extracts statistics from numeric columns
- **Vector Database Integration**: Seamless integration with Qdrant for efficient vector storage and retrieval
- **Amazon Bedrock Integration**: 
  - Amazon Titan for embeddings
  - Claude 3 Sonnet for advanced RAG capabilities
- **REST API**: Comprehensive API for document uploading, processing, and querying
- **Comparison Engine**: Compare results from different chunking strategies with detailed analysis

## Architecture

The system follows this workflow:
1. Documents (PDF/HTML) are uploaded and processed using specialized parsers
2. Documents are chunked using either recursive or semantic chunking
3. Chunks are embedded using Amazon Titan and stored in Qdrant
4. Queries are processed against both collections with Claude 3
5. Results are compared with detailed analysis of each strategy's performance

## Installation

### Prerequisites

- Python 3.9+
- Docker (for Qdrant)
- AWS account with Bedrock access (Claude 3 Sonnet and Titan)
- Poppler (optional, for enhanced PDF processing)

### Setup

1. Clone the repository and set up the environment:
   ```bash
   git clone https://github.com/yourusername/semantic_chunking.git
   cd semantic_chunking
   ./setup.sh  # This will handle all setup steps automatically
   ```

   Or follow these manual steps:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Configure AWS credentials in `.env`:
   ```
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_REGION=your_region
   CLAUDE_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
   ```

3. Start the application:
   ```bash
   uvicorn app:app --reload
   ```

## API Usage

### Document Processing

1. Upload a document:
```bash
curl -X POST -F "file=@your_document.pdf" http://localhost:8000/upload
```

2. Process with both chunking strategies:
```bash
# Recursive chunking
curl -X POST -F "file_path=uploads/your_document.pdf" -F "collection_name=recursive" http://localhost:8000/process/recursive

# Semantic chunking
curl -X POST -F "file_path=uploads/your_document.pdf" -F "collection_name=semantic" http://localhost:8000/process/semantic
```

### Querying and Comparison

The query endpoint now provides comprehensive comparison between chunking strategies:

```bash
curl -X POST -H "Content-Type: application/json" -d '{
    "query": "your question here",
    "recursive_collection": "recursive",
    "semantic_collection": "semantic"
}' http://localhost:8000/api/query
```

Response includes:
- Generated answers from both chunking methods
- Detailed comparison analysis
- Vector similarity metrics
- Content structure analysis
- Metadata richness comparison

## Example Queries

### Financial Analysis
```bash
curl -X POST -H "Content-Type: application/json" -d '{
    "query": "What was the quarterly revenue trend and its key drivers?",
    "recursive_collection": "recursive",
    "semantic_collection": "semantic"
}' http://localhost:8000/api/query
```

Expected differences:
- Semantic: Provides structured financial data, preserves table relationships, includes trend analysis
- Recursive: May miss relationships between numbers and explanatory text

### Document Structure
```bash
curl -X POST -H "Content-Type: application/json" -d '{
    "query": "Summarize the key risks and their potential impact on financials",
    "recursive_collection": "recursive",
    "semantic_collection": "semantic"
}' http://localhost:8000/api/query
```

Expected differences:
- Semantic: Maintains section context, links risks to financial impacts
- Recursive: May miss cross-references between sections

### Table Analysis
```bash
curl -X POST -H "Content-Type: application/json" -d '{
    "query": "Compare the profit margins across business segments",
    "recursive_collection": "recursive",
    "semantic_collection": "semantic"
}' http://localhost:8000/api/query
```

Expected differences:
- Semantic: Preserves table structure, enables accurate calculations
- Recursive: May split tables incorrectly, leading to inaccurate calculations

### Mixed Content
```bash
curl -X POST -H "Content-Type: application/json" -d '{
    "query": "How do market trends affect our product performance?",
    "recursive_collection": "recursive",
    "semantic_collection": "semantic"
}' http://localhost:8000/api/query
```

Expected differences:
- Semantic: Integrates market analysis text with performance tables
- Recursive: May miss connections between narrative and data

## Response Analysis

The query response includes:

```json
{
    "query": "your question",
    "recursive": {
        "collection": "recursive",
        "answer": "generated answer",
        "chunks": [{"content": "...", "metadata": {...}}]
    },
    "semantic": {
        "collection": "semantic",
        "answer": "generated answer",
        "chunks": [{"content": "...", "metadata": {...}}]
    },
    "analysis": {
        "rag_comparison": "detailed comparison",
        "vector_comparison": "similarity analysis"
    }
}
```

## Project Structure

```
semantic_chunking/
├── api/                      # API routes and schemas
│   ├── routes.py             # API endpoints
│   └── schemas.py            # Pydantic models
├── src/                      # Source code
│   ├── chunking/             # Chunking strategies
│   │   ├── recursive.py      # Recursive chunking implementation
│   │   └── semantic.py       # Semantic chunking implementation
│   ├── embeddings/           # Embedding models
│   │   └── titan.py          # Amazon Titan embeddings
│   ├── rag/                  # RAG implementation
│   │   └── query.py          # Query implementation
│   ├── storage/              # Vector storage
│   │   └── qdrant.py         # Qdrant storage implementation
│   └── utils/                # Utility functions
│       ├── config.py         # Configuration
│       ├── parser.py         # PDF parsing
│       └── pdf.py            # PDF utilities
├── uploads/                  # Upload directory
├── app.py                    # Main application
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for the RAG framework
- [Unstructured](https://github.com/Unstructured-IO/unstructured) for PDF processing
- [Qdrant](https://github.com/qdrant/qdrant) for vector storage
- [FastAPI](https://github.com/tiangolo/fastapi) for the API framework
- [Amazon Bedrock](https://aws.amazon.com/bedrock/) for embeddings and LLM capabilities 