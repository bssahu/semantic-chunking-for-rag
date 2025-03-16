#!/usr/bin/env python3
"""
Test script for the Semantic Document Chunking and RAG API.
This script tests the API endpoints to help diagnose issues.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API base URL
BASE_URL = "http://localhost:8000"

def test_upload(pdf_path):
    """Test the file upload endpoint."""
    logger.info(f"Testing file upload: {pdf_path}")
    
    url = f"{BASE_URL}/api/upload"
    
    with open(pdf_path, "rb") as f:
        files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        result = response.json()
        logger.info(f"Upload successful: {result['file_path']}")
        return result["file_path"]
    else:
        logger.error(f"Upload failed: {response.status_code} - {response.text}")
        return None

def test_recursive_chunking(pdf_path):
    """Test the recursive chunking endpoint."""
    logger.info(f"Testing recursive chunking: {pdf_path}")
    
    url = f"{BASE_URL}/api/chunk/recursive"
    
    data = {
        "pdf_path": pdf_path,
        "collection_name": None
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        logger.info(f"Recursive chunking successful: {result['num_chunks']} chunks created")
        return result
    else:
        logger.error(f"Recursive chunking failed: {response.status_code} - {response.text}")
        return None

def test_semantic_chunking(pdf_path):
    """Test the semantic chunking endpoint."""
    logger.info(f"Testing semantic chunking: {pdf_path}")
    
    url = f"{BASE_URL}/api/chunk/semantic"
    
    data = {
        "pdf_path": pdf_path,
        "collection_name": None
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        logger.info(f"Semantic chunking successful: {result['num_chunks']} chunks created")
        return result
    else:
        logger.error(f"Semantic chunking failed: {response.status_code} - {response.text}")
        return None

def test_rag_query(query):
    """Test the RAG query endpoint."""
    logger.info(f"Testing RAG query: {query}")
    
    url = f"{BASE_URL}/api/rag/query"
    
    data = {
        "query": query
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        logger.info("RAG query successful")
        return result
    else:
        logger.error(f"RAG query failed: {response.status_code} - {response.text}")
        return None

def main():
    """Main function to test the API endpoints."""
    parser = argparse.ArgumentParser(description="Test the Semantic Document Chunking and RAG API")
    parser.add_argument("--pdf", type=str, default="enhanced_financial_report.pdf", help="Path to the PDF file")
    parser.add_argument("--query", type=str, default="What are the key financial metrics mentioned in the report?", help="Query to search for")
    parser.add_argument("--skip-upload", action="store_true", help="Skip the upload step and use an existing file")
    parser.add_argument("--skip-chunking", action="store_true", help="Skip the chunking steps")
    args = parser.parse_args()
    
    # Check if the server is running
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code != 200:
            logger.error(f"Server is not responding correctly: {response.status_code} - {response.text}")
            return 1
    except requests.exceptions.ConnectionError:
        logger.error(f"Could not connect to server at {BASE_URL}")
        logger.error("Make sure the server is running with: python app.py")
        return 1
    
    # Test file upload
    if args.skip_upload:
        logger.info("Skipping upload step")
        pdf_path = args.pdf
        if not os.path.exists(pdf_path) and not pdf_path.startswith("uploads/"):
            pdf_path = f"uploads/{os.path.basename(pdf_path)}"
    else:
        pdf_path = test_upload(args.pdf)
        if not pdf_path:
            return 1
    
    # Test chunking
    if not args.skip_chunking:
        # Test recursive chunking
        recursive_result = test_recursive_chunking(pdf_path)
        if not recursive_result:
            logger.error("Recursive chunking failed, but continuing with other tests")
        
        # Test semantic chunking
        semantic_result = test_semantic_chunking(pdf_path)
        if not semantic_result:
            logger.error("Semantic chunking failed, but continuing with other tests")
    else:
        logger.info("Skipping chunking steps")
    
    # Test RAG query
    rag_result = test_rag_query(args.query)
    if not rag_result:
        return 1
    
    # Print RAG results
    logger.info("\nRAG Query Results:")
    logger.info(f"Query: {rag_result['query']}")
    
    logger.info("\nRecursive Chunking Result:")
    logger.info("-" * 80)
    logger.info(rag_result['recursive']['answer'])
    
    logger.info("\nSemantic Chunking Result:")
    logger.info("-" * 80)
    logger.info(rag_result['semantic']['answer'])
    
    logger.info("\nComparison:")
    logger.info("-" * 80)
    logger.info(rag_result['comparison'])
    
    logger.info("\nAll tests completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 