import os
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import pypdf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import unstructured, but provide fallback if it fails
try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.documents.elements import (
        Element, 
        Title, 
        NarrativeText, 
        Table, 
        ListItem,
        Image
    )
    UNSTRUCTURED_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import unstructured: {str(e)}")
    logger.warning("Falling back to basic PDF processing without structured elements")
    UNSTRUCTURED_AVAILABLE = False

from src.utils.parser import parse_pdf

def extract_text_with_pypdf(pdf_path: str) -> str:
    """
    Extract text from PDF using PyPDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    with open(pdf_path, "rb") as file:
        pdf_reader = pypdf.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def extract_structured_elements(pdf_path: str) -> List[Dict]:
    """
    Extract structured elements from PDF using Unstructured.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of structured elements as dictionaries
    """
    if not UNSTRUCTURED_AVAILABLE:
        # Fallback to basic extraction if unstructured is not available
        text = extract_text_with_pypdf(pdf_path)
        
        # Create a simple structure with the extracted text
        elements = [{
            "type": "NarrativeText",
            "text": text,
            "metadata": {"page_number": 1}
        }]
        
        return elements
    
    try:
        # Use unstructured for advanced extraction
        raw_elements = partition_pdf(
            pdf_path,
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3000,
            combine_text_under_n_chars=2000,
        )
        
        # Convert elements to dictionaries
        elements = [element_to_dict(element) for element in raw_elements]
        return elements
    
    except Exception as e:
        logger.error(f"Error using unstructured: {str(e)}")
        logger.info("Falling back to basic PDF processing")
        
        # Fallback to basic extraction
        text = extract_text_with_pypdf(pdf_path)
        
        # Create a simple structure with the extracted text
        elements = [{
            "type": "NarrativeText",
            "text": text,
            "metadata": {"page_number": 1}
        }]
        
        return elements

def element_to_dict(element: Element) -> Dict:
    """
    Convert an Unstructured element to a dictionary.
    
    Args:
        element: Unstructured element
        
    Returns:
        Dictionary representation of the element
    """
    if not UNSTRUCTURED_AVAILABLE:
        return {}
    
    element_type = type(element).__name__
    
    result = {
        "type": element_type,
        "text": str(element),
        "metadata": {}
    }
    
    # Add element-specific metadata
    if hasattr(element, "metadata"):
        result["metadata"] = element.metadata.to_dict() if hasattr(element.metadata, "to_dict") else element.metadata
    
    # Add page number if available
    if hasattr(element, "metadata") and hasattr(element.metadata, "page_number"):
        result["metadata"]["page_number"] = element.metadata.page_number
    
    # Add table data if it's a table
    if element_type == "Table" and hasattr(element, "metadata") and hasattr(element.metadata, "text_as_html"):
        result["metadata"]["html"] = element.metadata.text_as_html
    
    return result

def process_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Process a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of structured elements
    """
    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Parse PDF
    elements = parse_pdf(file_path)
    
    return elements 