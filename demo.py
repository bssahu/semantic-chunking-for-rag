import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for AWS credentials
if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
    print("Error: AWS credentials not found in .env file")
    print("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your .env file")
    print("Example:")
    print("AWS_ACCESS_KEY_ID=your_access_key")
    print("AWS_SECRET_ACCESS_KEY=your_secret_key")
    print("AWS_REGION=us-east-1")
    sys.exit(1)

from src.chunking.recursive import RecursiveChunker
from src.chunking.semantic import SemanticChunker
from src.rag.query import RAGQuery
from src.utils.pdf import process_pdf

def main():
    """
    Demo script to demonstrate the application's functionality.
    """
    parser = argparse.ArgumentParser(description="Semantic Document Chunking and RAG Demo")
    parser.add_argument("--pdf", type=str, default="enhanced_financial_report.pdf", help="Path to the PDF file")
    parser.add_argument("--query", type=str, default="What are the key financial metrics mentioned in the report?", help="Query to search for")
    args = parser.parse_args()
    
    pdf_path = args.pdf
    query = args.query
    
    if not Path(pdf_path).exists():
        print(f"Error: PDF file '{pdf_path}' not found")
        return
    
    print(f"Processing PDF: {pdf_path}")
    print("=" * 80)
    
    try:
        # Process PDF
        pdf_data = process_pdf(pdf_path)
        print(f"Extracted {len(pdf_data['raw_text'])} characters of raw text")
        print(f"Extracted {len(pdf_data['structured_elements'])} structured elements")
        print("=" * 80)
        
        # Process with recursive chunking
        print("Processing with recursive chunking...")
        recursive_chunker = RecursiveChunker()
        recursive_chunks = recursive_chunker.process_and_store(pdf_data["raw_text"])
        print(f"Created {len(recursive_chunks)} recursive chunks")
        print("=" * 80)
        
        # Process with semantic chunking
        print("Processing with semantic chunking...")
        semantic_chunker = SemanticChunker()
        semantic_chunks = semantic_chunker.process_and_store(pdf_data["structured_elements"])
        print(f"Created {len(semantic_chunks)} semantic chunks")
        print("=" * 80)
        
        # Query both collections
        print(f"Querying: {query}")
        rag = RAGQuery()
        result = rag.query(query)
        
        print("\nRecursive Chunking Result:")
        print("-" * 80)
        print(result["recursive"]["answer"])
        print("\nSemantic Chunking Result:")
        print("-" * 80)
        print(result["semantic"]["answer"])
        print("\nComparison:")
        print("-" * 80)
        print(result["comparison"])
    
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your AWS credentials are correctly set in the .env file")
        print("2. Ensure the Qdrant database is running (docker ps should show the qdrant container)")
        print("3. Check that you have all the required dependencies installed")
        print("4. If you get import errors, try reinstalling the requirements with 'pip install -r requirements.txt'")

if __name__ == "__main__":
    main() 