import logging
from typing import List, Dict, Any
import os
import traceback

from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    Element, 
    Title, 
    NarrativeText, 
    Table, 
    ListItem
)

# Configure logging
logger = logging.getLogger(__name__)

# Try to import PyPDF for fallback
try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    logger.warning("PyPDF not available for fallback PDF parsing")
    PYPDF_AVAILABLE = False

def extract_text_with_pypdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from PDF using PyPDF as a fallback method.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of structured elements
    """
    logger.info(f"Using PyPDF fallback for PDF parsing: {file_path}")
    
    try:
        # Open the PDF file
        with open(file_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)
            
            # Extract text from each page
            elements = []
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                
                if text.strip():  # Only add non-empty pages
                    elements.append({
                        "type": "NarrativeText",
                        "text": text,
                        "metadata": {
                            "page_number": page_num + 1
                        }
                    })
            
            logger.info(f"Extracted {len(elements)} pages with PyPDF fallback")
            return elements
    except Exception as e:
        logger.error(f"Error in PyPDF fallback: {str(e)}")
        # Return a single element with error message
        return [{
            "type": "NarrativeText",
            "text": f"Error extracting text from PDF: {str(e)}",
            "metadata": {"page_number": 1}
        }]

def parse_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse a PDF file into structured elements.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of structured elements
    """
    logger.info(f"Parsing PDF: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Parse PDF
    try:
        # Try using unstructured first
        elements = partition_pdf(
            filename=file_path,
            extract_images_in_pdf=False,
            infer_table_structure=True,
            strategy="hi_res"
        )
        
        # Convert elements to dictionaries
        result = []
        for element in elements:
            # Get element type
            element_type = element.__class__.__name__
            
            # Get element text
            text = str(element)
            
            # Get element metadata
            metadata = {}
            if hasattr(element, "metadata"):
                metadata = element.metadata.to_dict()
            
            # Add to result
            result.append({
                "type": element_type,
                "text": text,
                "metadata": metadata
            })
        
        logger.info(f"Parsed {len(result)} elements from PDF using unstructured")
        return result
        
    except Exception as e:
        logger.error(f"Error parsing PDF with unstructured: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Try fallback with PyPDF if available
        if PYPDF_AVAILABLE:
            logger.info("Attempting fallback with PyPDF")
            return extract_text_with_pypdf(file_path)
        else:
            logger.error("No fallback method available")
            raise Exception(f"Error parsing PDF: {str(e)}") 