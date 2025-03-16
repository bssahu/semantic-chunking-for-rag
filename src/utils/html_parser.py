"""
HTML Parser for extracting structured content from HTML documents.
Handles complex tables, nested tables, headers, and body text.
"""

import logging
import json
import pandas as pd
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Tuple, Optional, Union
from langchain.schema import Document

logger = logging.getLogger(__name__)

class HTMLParser:
    """Parser for HTML documents with enhanced table handling."""
    
    def __init__(self):
        """Initialize the HTML parser."""
        self.soup = None
        
    def parse_html(self, html_content: str) -> Dict[str, Any]:
        """
        Parse HTML content and extract structured elements.
        
        Args:
            html_content: HTML content as string
            
        Returns:
            Dictionary containing extracted elements
        """
        self.soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract document structure
        result = {
            "title": self._extract_title(),
            "headers": self._extract_headers(),
            "paragraphs": self._extract_paragraphs(),
            "tables": self._extract_tables(),
            "lists": self._extract_lists()
        }
        
        return result
    
    def _extract_title(self) -> str:
        """Extract document title."""
        title_tag = self.soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)
        
        # Try to find the first h1 as fallback
        h1_tag = self.soup.find('h1')
        if h1_tag:
            return h1_tag.get_text(strip=True)
            
        return "Untitled Document"
    
    def _extract_headers(self) -> List[Dict[str, str]]:
        """Extract all headers with their hierarchy level."""
        headers = []
        for level in range(1, 7):  # h1 to h6
            for header in self.soup.find_all(f'h{level}'):
                headers.append({
                    "level": level,
                    "text": header.get_text(strip=True)
                })
        return headers
    
    def _extract_paragraphs(self) -> List[str]:
        """Extract all paragraphs."""
        paragraphs = []
        for p in self.soup.find_all('p'):
            text = p.get_text(strip=True)
            if text:  # Only add non-empty paragraphs
                paragraphs.append(text)
        return paragraphs
    
    def _extract_lists(self) -> List[Dict[str, Any]]:
        """Extract ordered and unordered lists."""
        lists = []
        
        # Process unordered lists
        for ul in self.soup.find_all('ul'):
            items = [li.get_text(strip=True) for li in ul.find_all('li')]
            lists.append({
                "type": "unordered",
                "items": items
            })
        
        # Process ordered lists
        for ol in self.soup.find_all('ol'):
            items = [li.get_text(strip=True) for li in ol.find_all('li')]
            lists.append({
                "type": "ordered",
                "items": items
            })
            
        return lists
    
    def _extract_tables(self) -> List[Dict[str, Any]]:
        """Extract all tables with enhanced structure handling."""
        tables = []
        
        for idx, table in enumerate(self.soup.find_all('table')):
            table_data = self._process_table(table, idx)
            if table_data:
                tables.append(table_data)
                
        return tables
    
    def _process_table(self, table, table_idx: int) -> Dict[str, Any]:
        """
        Process a table element and extract its structure.
        
        Args:
            table: BeautifulSoup table element
            table_idx: Index of the table in the document
            
        Returns:
            Dictionary containing table structure and data
        """
        # Extract caption if available
        caption = table.find('caption')
        caption_text = caption.get_text(strip=True) if caption else f"Table {table_idx + 1}"
        
        # Extract headers
        headers = []
        thead = table.find('thead')
        if thead:
            header_rows = thead.find_all('tr')
            for row in header_rows:
                header_cells = []
                for cell in row.find_all(['th', 'td']):
                    # Handle colspan and rowspan
                    colspan = int(cell.get('colspan', 1))
                    rowspan = int(cell.get('rowspan', 1))
                    header_cells.append({
                        "text": cell.get_text(strip=True),
                        "colspan": colspan,
                        "rowspan": rowspan
                    })
                headers.append(header_cells)
        else:
            # Try to use the first row as header if thead is not explicitly defined
            first_row = table.find('tr')
            if first_row:
                header_cells = []
                for cell in first_row.find_all(['th', 'td']):
                    colspan = int(cell.get('colspan', 1))
                    rowspan = int(cell.get('rowspan', 1))
                    header_cells.append({
                        "text": cell.get_text(strip=True),
                        "colspan": colspan,
                        "rowspan": rowspan
                    })
                headers.append(header_cells)
        
        # Extract rows (excluding header rows if they were in thead)
        rows = []
        tbody = table.find('tbody') or table
        
        # If we used the first row as header and there's no thead, skip it in the body
        start_idx = 1 if not table.find('thead') and headers and tbody == table else 0
        
        for row in tbody.find_all('tr')[start_idx:]:
            row_cells = []
            for cell in row.find_all(['td', 'th']):
                # Handle colspan and rowspan
                colspan = int(cell.get('colspan', 1))
                rowspan = int(cell.get('rowspan', 1))
                
                # Check for nested tables
                nested_tables = []
                for nested_idx, nested_table in enumerate(cell.find_all('table')):
                    nested_table_data = self._process_table(nested_table, nested_idx)
                    if nested_table_data:
                        nested_tables.append(nested_table_data)
                
                cell_data = {
                    "text": cell.get_text(strip=True),
                    "colspan": colspan,
                    "rowspan": rowspan
                }
                
                if nested_tables:
                    cell_data["nested_tables"] = nested_tables
                    
                row_cells.append(cell_data)
            
            if row_cells:  # Only add non-empty rows
                rows.append(row_cells)
        
        # Create a simplified version for pandas DataFrame
        simple_headers = []
        if headers:
            # Use the last row of headers as column names
            simple_headers = [cell["text"] for cell in headers[-1]]
        
        simple_rows = []
        for row in rows:
            simple_row = [cell["text"] for cell in row]
            simple_rows.append(simple_row)
        
        # Create pandas DataFrame if possible
        df = None
        if simple_headers and simple_rows:
            try:
                # Adjust if rows have different lengths than headers
                for row in simple_rows:
                    while len(row) < len(simple_headers):
                        row.append("")
                    if len(row) > len(simple_headers):
                        row = row[:len(simple_headers)]
                
                df = pd.DataFrame(simple_rows, columns=simple_headers)
                
                # Try to convert numeric columns
                for col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except (ValueError, TypeError):
                        pass
            except Exception as e:
                logger.warning(f"Failed to create DataFrame for table: {e}")
        
        # Generate statistics for numeric columns
        stats = {}
        if df is not None:
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    stats[col] = {
                        "min": df[col].min(),
                        "max": df[col].max(),
                        "mean": df[col].mean(),
                        "median": df[col].median(),
                        "sum": df[col].sum()
                    }
        
        # Create a readable text representation
        text_representation = self._table_to_text(caption_text, simple_headers, simple_rows)
        
        return {
            "caption": caption_text,
            "headers": headers,
            "rows": rows,
            "dataframe": df.to_dict('records') if df is not None else None,
            "statistics": stats,
            "text_representation": text_representation
        }
    
    def _table_to_text(self, caption: str, headers: List[str], rows: List[List[str]]) -> str:
        """Convert table to a readable text representation."""
        text = f"{caption}\n\n"
        
        if headers:
            text += " | ".join(headers) + "\n"
            text += "-" * (sum(len(h) for h in headers) + 3 * (len(headers) - 1)) + "\n"
        
        for row in rows:
            text += " | ".join(row) + "\n"
            
        return text
    
    def create_documents(self, html_content: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Create LangChain Document objects from HTML content.
        
        Args:
            html_content: HTML content as string
            metadata: Additional metadata to include
            
        Returns:
            List of Document objects
        """
        if metadata is None:
            metadata = {}
            
        parsed_content = self.parse_html(html_content)
        documents = []
        
        # Create document for title and headers
        title_text = parsed_content["title"]
        headers_text = "\n".join([f"{'#' * h['level']} {h['text']}" for h in parsed_content["headers"]])
        
        if title_text or headers_text:
            title_doc = Document(
                page_content=f"{title_text}\n\n{headers_text}",
                metadata={
                    **metadata,
                    "source": "title_and_headers",
                    "document_type": "header"
                }
            )
            documents.append(title_doc)
        
        # Create document for each paragraph
        for i, paragraph in enumerate(parsed_content["paragraphs"]):
            para_doc = Document(
                page_content=paragraph,
                metadata={
                    **metadata,
                    "source": f"paragraph_{i}",
                    "document_type": "text"
                }
            )
            documents.append(para_doc)
        
        # Create document for each list
        for i, list_item in enumerate(parsed_content["lists"]):
            list_type = list_item["type"]
            items = list_item["items"]
            
            if list_type == "ordered":
                list_text = "\n".join([f"{j+1}. {item}" for j, item in enumerate(items)])
            else:
                list_text = "\n".join([f"• {item}" for item in items])
                
            list_doc = Document(
                page_content=list_text,
                metadata={
                    **metadata,
                    "source": f"list_{i}",
                    "document_type": "list",
                    "list_type": list_type
                }
            )
            documents.append(list_doc)
        
        # Create documents for each table
        for i, table in enumerate(parsed_content["tables"]):
            # Document with text representation
            table_doc = Document(
                page_content=table["text_representation"],
                metadata={
                    **metadata,
                    "source": f"table_{i}",
                    "document_type": "table",
                    "caption": table["caption"]
                }
            )
            documents.append(table_doc)
            
            # Document with structured data
            structured_data = {
                "caption": table["caption"],
                "data": table["dataframe"] if table["dataframe"] else [],
                "statistics": table["statistics"]
            }
            
            table_structured_doc = Document(
                page_content=f"Structured data for table: {table['caption']}",
                metadata={
                    **metadata,
                    "source": f"table_{i}_structured",
                    "document_type": "table_structured",
                    "caption": table["caption"],
                    "structured_data": json.dumps(structured_data)
                }
            )
            documents.append(table_structured_doc)
            
            # If there are statistics, create a special document for them
            if table["statistics"]:
                stats_text = f"Statistics for table: {table['caption']}\n\n"
                for col, col_stats in table["statistics"].items():
                    stats_text += f"Column: {col}\n"
                    for stat_name, stat_value in col_stats.items():
                        stats_text += f"  {stat_name}: {stat_value}\n"
                
                stats_doc = Document(
                    page_content=stats_text,
                    metadata={
                        **metadata,
                        "source": f"table_{i}_statistics",
                        "document_type": "table_statistics",
                        "caption": table["caption"]
                    }
                )
                documents.append(stats_doc)
        
        return documents


def parse_html_file(file_path: str, metadata: Dict[str, Any] = None) -> List[Document]:
    """
    Parse an HTML file and return a list of Document objects.
    
    Args:
        file_path: Path to the HTML file
        metadata: Additional metadata to include
        
    Returns:
        List of Document objects
    """
    if metadata is None:
        metadata = {"source": file_path}
    else:
        metadata = {**metadata, "source": file_path}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        parser = HTMLParser()
        return parser.create_documents(html_content, metadata)
    except Exception as e:
        logger.error(f"Error parsing HTML file {file_path}: {e}")
        return []


def parse_html_content(html_content: str, metadata: Dict[str, Any] = None) -> List[Document]:
    """
    Parse HTML content and return a list of Document objects.
    
    Args:
        html_content: HTML content as string
        metadata: Additional metadata to include
        
    Returns:
        List of Document objects
    """
    if metadata is None:
        metadata = {"source": "html_content"}
    
    try:
        parser = HTMLParser()
        return parser.create_documents(html_content, metadata)
    except Exception as e:
        logger.error(f"Error parsing HTML content: {e}")
        return [] 