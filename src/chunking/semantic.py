from typing import Dict, List, Optional, Union, Any
import re
import json
import logging
import pandas as pd
from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from unstructured.documents.elements import (
    Element, 
    Title, 
    NarrativeText, 
    Table, 
    ListItem
)

from src.embeddings.titan import TitanEmbeddings
from src.storage.qdrant import QdrantStorage
from src.utils.config import (
    SEMANTIC_COLLECTION_NAME,
    SEMANTIC_CHUNK_SIZE,
    SEMANTIC_CHUNK_OVERLAP,
    load_config
)

# Configure logging
logger = logging.getLogger(__name__)

class SemanticChunker:
    """
    Semantic chunking implementation with enhanced table handling.
    """
    
    def __init__(self):
        """Initialize the semantic chunker with configuration."""
        self.chunk_size = int(load_config("SEMANTIC_CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(load_config("SEMANTIC_CHUNK_OVERLAP", "200"))
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        self.embeddings = TitanEmbeddings()
        self.storage = QdrantStorage()
    
    def extract_table_from_html(self, table_element) -> Dict:
        """
        Extract table data from HTML and convert to structured format.
        
        Args:
            table_element: BeautifulSoup table element
            
        Returns:
            Dictionary with table data
        """
        try:
            # Extract headers
            headers = []
            header_row = table_element.find('thead')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
            
            # Extract rows
            rows = []
            for tr in table_element.find_all('tr'):
                row = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                if row and row != headers:  # Skip header row if we already got it
                    rows.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=headers if headers else None)
            
            # Convert to structured format
            table_data = {
                "headers": headers,
                "rows": rows,
                "records": json.loads(df.to_json(orient='records'))
            }
            
            return table_data
            
        except Exception as e:
            logger.warning(f"Failed to extract table data: {e}")
            return {
                "headers": headers if 'headers' in locals() else [],
                "rows": rows if 'rows' in locals() else [],
                "error": str(e)
            }
    
    def process_table(self, table_element, metadata: Dict = None) -> Dict[str, Any]:
        """
        Create a document from a table element.
        
        Args:
            table_element: BeautifulSoup table element
            metadata: Optional metadata
            
        Returns:
            Document with table data
        """
        try:
            # Extract table data
            table_data = self.extract_table_from_html(table_element)
            
            # Create readable text representation
            text_parts = []
            
            # Add headers
            if table_data["headers"]:
                text_parts.append("Table Headers: " + " | ".join(table_data["headers"]))
            
            # Add rows
            text_parts.append("\nTable Data:")
            max_cols = max(len(row) for row in table_data["rows"]) if table_data["rows"] else 0
            
            # Pad rows to have equal length
            padded_rows = []
            for row in table_data["rows"]:
                padded_row = row + [""] * (max_cols - len(row))
                padded_rows.append(padded_row)
            
            # Create DataFrame with padded rows
            try:
                if table_data["headers"] and len(table_data["headers"]) == max_cols:
                    df = pd.DataFrame(padded_rows, columns=table_data["headers"])
                else:
                    # Create default column names if headers don't match
                    columns = [f"Column_{i+1}" for i in range(max_cols)]
                    df = pd.DataFrame(padded_rows, columns=columns)
                
                # Add DataFrame records to table data
                table_data["records"] = json.loads(df.to_json(orient='records'))
                
                # Add basic statistics for numeric columns
                table_data["statistics"] = {}
                for column in df.columns:
                    try:
                        numeric_series = pd.to_numeric(df[column], errors='coerce')
                        if not numeric_series.isna().all():
                            table_data["statistics"][column] = {
                                "min": float(numeric_series.min()),
                                "max": float(numeric_series.max()),
                                "mean": float(numeric_series.mean()),
                                "sum": float(numeric_series.sum())
                            }
                    except:
                        continue
                
            except Exception as e:
                logger.warning(f"Error creating DataFrame: {e}")
            
            # Format rows for text representation
            for row in padded_rows:
                text_parts.append(" | ".join(str(cell) for cell in row))
            
            return {
                "content": "\n".join(text_parts),
                "metadata": {
                    **(metadata or {}),
                    "type": "table",
                    "chunking_type": "semantic",
                    "table_data": table_data
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing table: {e}")
            # Return a minimal document if table processing fails
            return {
                "content": table_element.get_text(),
                "metadata": {
                    **(metadata or {}),
                    "type": "table",
                    "chunking_type": "semantic",
                    "error": str(e)
                }
            }
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk documents using semantic understanding.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        try:
            chunked_documents = []
            
            for doc in documents:
                # Handle both Document objects and dictionaries
                if isinstance(doc, Document):
                    content = doc.page_content
                    metadata = doc.metadata
                else:
                    content = doc["content"]
                    metadata = doc.get("metadata", {})
                
                # Check if content is HTML
                if metadata.get("content_type") == "text/html" or "<html" in content.lower():
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Process tables
                    tables = soup.find_all('table')
                    logger.info(f"Found {len(tables)} tables in document")
                    for i, table in enumerate(tables):
                        try:
                            table_doc = self.process_table(table, {
                                **metadata,
                                "table_index": i,
                                "total_tables": len(tables)
                            })
                            chunked_documents.append(table_doc)
                        except Exception as e:
                            logger.warning(f"Error processing table {i}: {e}")
                    
                    # Process text elements
                    text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
                    logger.info(f"Found {len(text_elements)} text elements in document")
                    
                    # Group text elements by section
                    current_section = {"title": "", "content": [], "type": ""}
                    sections = []
                    
                    for element in text_elements:
                        text = element.get_text(strip=True)
                        if not text:  # Skip empty elements
                            continue
                            
                        # Check if this is a heading
                        if element.name.startswith('h'):
                            # Save previous section if it exists
                            if current_section["content"]:
                                sections.append(current_section.copy())
                            
                            # Start new section
                            current_section = {
                                "title": text,
                                "content": [text],
                                "type": "section",
                                "level": int(element.name[1])
                            }
                        else:
                            # Add to current section
                            current_section["content"].append(text)
                    
                    # Add the last section
                    if current_section["content"]:
                        sections.append(current_section)
                    
                    # Process each section
                    for i, section in enumerate(sections):
                        section_text = "\n".join(section["content"])
                        
                        # Create chunks from section
                        section_chunks = self.text_splitter.create_documents(
                            texts=[section_text],
                            metadatas=[{
                                **metadata,
                                "type": "section",
                                "section_title": section["title"],
                                "section_index": i,
                                "total_sections": len(sections),
                                "chunking_type": "semantic"
                            }]
                        )
                        
                        # Convert Document objects to dictionaries
                        for chunk in section_chunks:
                            chunked_documents.append({
                                "content": chunk.page_content,
                                "metadata": chunk.metadata
                            })
                else:
                    # For non-HTML content, create semantic chunks
                    chunks = self.text_splitter.create_documents(
                        texts=[content],
                        metadatas=[{
                            **metadata,
                            "chunking_type": "semantic"
                        }]
                    )
                    
                    # Convert Document objects to dictionaries
                    for chunk in chunks:
                        chunked_documents.append({
                            "content": chunk.page_content,
                            "metadata": chunk.metadata
                        })
            
            logger.info(f"Created {len(chunked_documents)} chunks from {len(documents)} documents")
            return chunked_documents
            
        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            raise
    
    def create_table_chunks(self, table_data: Dict, metadata: Dict) -> List[Document]:
        """
        Create multiple specialized chunks from table data for better retrieval.
        
        Args:
            table_data: Structured table data
            metadata: Base metadata for the table
            
        Returns:
            List of document chunks representing different aspects of the table
        """
        chunks = []
        
        # Base metadata with table identifier
        base_metadata = {
            **metadata,
            "type": "table",
            "chunking_strategy": "semantic"
        }
        
        # 1. Create a chunk for the full table in human-readable format
        readable_content = f"TABLE:\n"
        
        # Add headers
        if "headers" in table_data and table_data["headers"]:
            readable_content += " | ".join(table_data["headers"]) + "\n"
            readable_content += "-" * (len(readable_content) - 1) + "\n"
        
        # Add rows
        if "rows" in table_data and table_data["rows"]:
            for row in table_data["rows"]:
                readable_content += " | ".join(row) + "\n"
        
        chunks.append(Document(
            page_content=readable_content,
            metadata={
                **base_metadata,
                "table_format": "readable",
                "table_purpose": "display"
            }
        ))
        
        # 2. Create a chunk with the JSON records for structured querying
        if "records" in table_data and table_data["records"]:
            # Store the raw JSON for direct access
            json_str = json.dumps(table_data["records"], indent=2)
            
            # Create a more structured representation for the content
            json_content = f"TABLE DATA (JSON FORMAT):\n{json_str}"
            
            chunks.append(Document(
                page_content=json_content,
                metadata={
                    **base_metadata,
                    "table_format": "json",
                    "table_purpose": "query",
                    "json_data": json_str,  # Store the raw JSON for direct access
                    "record_count": len(table_data["records"]),
                    "fields": list(table_data["records"][0].keys()) if table_data["records"] else []
                }
            ))
            
            # 3. Create specialized chunks for each column with numeric data
            if "columns" in table_data and table_data["columns"]:
                for col_name, col_values in table_data["columns"].items():
                    # Try to convert to numeric if possible
                    try:
                        numeric_values = [float(v) if v and v.strip() else 0 for v in col_values if v]
                        if numeric_values:
                            # Create a specialized chunk for this column
                            col_content = f"COLUMN DATA: {col_name}\n\n"
                            col_content += f"Values: {numeric_values}\n"
                            
                            # Calculate basic statistics
                            if len(numeric_values) > 0:
                                col_content += f"Min: {min(numeric_values)}\n"
                                col_content += f"Max: {max(numeric_values)}\n"
                                col_content += f"Average: {sum(numeric_values)/len(numeric_values)}\n"
                                col_content += f"Sum: {sum(numeric_values)}\n"
                            
                            chunks.append(Document(
                                page_content=col_content,
                                metadata={
                                    **base_metadata,
                                    "table_format": "column",
                                    "table_purpose": "analysis",
                                    "column_name": col_name,
                                    "is_numeric": True,
                                    "column_stats": {
                                        "min": min(numeric_values) if numeric_values else None,
                                        "max": max(numeric_values) if numeric_values else None,
                                        "avg": sum(numeric_values)/len(numeric_values) if numeric_values else None,
                                        "sum": sum(numeric_values) if numeric_values else None,
                                        "count": len(numeric_values)
                                    }
                                }
                            ))
                    except (ValueError, TypeError):
                        # Not numeric, create a text-based column chunk
                        if col_values and any(col_values):
                            col_content = f"COLUMN DATA: {col_name}\n\n"
                            col_content += f"Values: {col_values}\n"
                            col_content += f"Unique values: {len(set(v for v in col_values if v))}\n"
                            
                            chunks.append(Document(
                                page_content=col_content,
                                metadata={
                                    **base_metadata,
                                    "table_format": "column",
                                    "table_purpose": "analysis",
                                    "column_name": col_name,
                                    "is_numeric": False,
                                    "unique_values": list(set(v for v in col_values if v))[:10]  # Limit to 10 values
                                }
                            ))
        
        # 4. Create a chunk with statistical information if available
        if "statistics" in table_data and table_data["statistics"]:
            stats_content = "TABLE STATISTICS:\n"
            
            for col, stats in table_data["statistics"].items():
                stats_content += f"\n{col}:\n"
                for stat_name, stat_value in stats.items():
                    stats_content += f"  - {stat_name}: {stat_value}\n"
            
            chunks.append(Document(
                page_content=stats_content,
                metadata={
                    **base_metadata,
                    "table_format": "statistics",
                    "table_purpose": "analysis",
                    "statistics": table_data["statistics"]  # Store raw statistics in metadata
                }
            ))
        
        # 5. Create a semantic description of the table
        if "headers" in table_data and "rows" in table_data:
            description = f"This table contains {len(table_data['rows'])} rows and {len(table_data['headers'])} columns. "
            description += f"The columns are: {', '.join(table_data['headers'])}. "
            
            # Add information about numeric columns
            if "statistics" in table_data:
                numeric_cols = list(table_data["statistics"].keys())
                if numeric_cols:
                    description += f"The table contains numeric data in columns: {', '.join(numeric_cols)}. "
                    
                    # Add key statistics for important columns
                    for col, stats in table_data["statistics"].items():
                        if "sum" in stats and "mean" in stats:
                            description += f"The sum of {col} is {stats['sum']} with an average of {stats['mean']}. "
            
            chunks.append(Document(
                page_content=description,
                metadata={
                    **base_metadata,
                    "table_format": "description",
                    "table_purpose": "overview"
                }
            ))
            
        # 6. Create row-based chunks for large tables to improve retrieval
        if "records" in table_data and table_data["records"] and len(table_data["records"]) > 10:
            # Group rows into smaller chunks for better retrieval
            row_chunk_size = 5  # Number of rows per chunk
            for i in range(0, len(table_data["records"]), row_chunk_size):
                row_group = table_data["records"][i:i+row_chunk_size]
                
                # Create content for this group of rows
                row_content = f"TABLE ROWS {i+1} to {i+len(row_group)}:\n\n"
                
                # Add headers
                if "headers" in table_data and table_data["headers"]:
                    row_content += " | ".join(table_data["headers"]) + "\n"
                    row_content += "-" * len(row_content) + "\n"
                
                # Add the rows in this group
                for row_dict in row_group:
                    row_values = [str(row_dict.get(h, "")) for h in table_data["headers"]]
                    row_content += " | ".join(row_values) + "\n"
                
                # Add as JSON for structured access
                row_json = json.dumps(row_group, indent=2)
                row_content += f"\nJSON:\n{row_json}"
                
                chunks.append(Document(
                    page_content=row_content,
                    metadata={
                        **base_metadata,
                        "table_format": "row_group",
                        "table_purpose": "query",
                        "row_range": f"{i+1}-{i+len(row_group)}",
                        "json_data": row_json
                    }
                ))
        
        logger.info(f"Created {len(chunks)} specialized chunks for table {metadata.get('table_id', 'unknown')}")
        return chunks
    
    def process_table(self, table: Dict) -> List[Document]:
        """
        Process a table element into structured documents.
        
        Args:
            table: Table element dictionary
            
        Returns:
            List of documents with structured table data
        """
        # Extract table HTML if available
        html_content = table["metadata"].get("html", "")
        
        # Default metadata
        metadata = {
            "page_number": table["metadata"].get("page_number", 0),
            "table_id": f"table_{table['metadata'].get('page_number', 0)}_{id(table)}"
        }
        
        # If HTML is available, extract structured data
        if html_content:
            structured_data = self.extract_table_from_html(BeautifulSoup(html_content, 'html.parser'))
            
            # Create multiple specialized chunks from the table data
            return self.create_table_chunks(structured_data, metadata)
        else:
            # If no HTML, use the raw text
            content = f"TABLE:\n{table['text']}"
            
            return [Document(
                page_content=content,
                metadata={
                    **metadata,
                    "type": "table",
                    "chunking_strategy": "semantic",
                    "table_format": "text",
                    "table_purpose": "display"
                }
            )]
    
    def process_elements(self, elements: List[Dict]) -> List[Document]:
        """
        Process structured elements using semantic chunking.
        
        Args:
            elements: List of structured elements
            
        Returns:
            List of chunks with metadata
        """
        # Group elements by type
        grouped_elements = {}
        for element in elements:
            element_type = element["type"]
            if element_type not in grouped_elements:
                grouped_elements[element_type] = []
            grouped_elements[element_type].append(element)
        
        # Process each type of element differently
        chunks = []
        
        # Process tables separately to preserve structure
        if "Table" in grouped_elements:
            logger.info(f"Processing {len(grouped_elements['Table'])} tables")
            for table in grouped_elements["Table"]:
                # Create multiple specialized chunks for each table
                table_chunks = self.process_table(table)
                chunks.extend(table_chunks)
                logger.info(f"Created {len(table_chunks)} chunks for table")
        
        # Process titles and narrative text for semantic chunking
        text_elements = []
        if "Title" in grouped_elements:
            text_elements.extend(grouped_elements["Title"])
        if "NarrativeText" in grouped_elements:
            text_elements.extend(grouped_elements["NarrativeText"])
        if "ListItem" in grouped_elements:
            text_elements.extend(grouped_elements["ListItem"])
        
        # Sort text elements by page number and position
        text_elements.sort(key=lambda x: (
            x["metadata"].get("page_number", 0),
            x["metadata"].get("coordinates", {}).get("y", 0) if isinstance(x["metadata"].get("coordinates"), dict) else 0
        ))
        
        logger.info(f"Processing {len(text_elements)} text elements")
        
        # Process text elements by section
        current_section = {"title": "", "content": "", "page": 0}
        sections = []
        
        for element in text_elements:
            page_num = element["metadata"].get("page_number", 0)
            
            if element["type"] == "Title":
                # If we have content in the current section, save it
                if current_section["content"]:
                    sections.append(current_section.copy())
                
                # Start a new section
                current_section = {
                    "title": element["text"],
                    "content": element["text"],
                    "page": page_num
                }
            else:
                # Add content to current section
                if current_section["content"]:
                    current_section["content"] += f"\n\n{element['text']}"
                else:
                    current_section["content"] = element["text"]
                    current_section["page"] = page_num
        
        # Add the last section if it has content
        if current_section["content"]:
            sections.append(current_section)
        
        # If no sections were created (e.g., from fallback parser with no titles)
        # create a single section from all text elements
        if not sections and text_elements:
            all_text = "\n\n".join([element["text"] for element in text_elements])
            page_num = text_elements[0]["metadata"].get("page_number", 0) if text_elements else 0
            
            sections.append({
                "title": "Document",
                "content": all_text,
                "page": page_num
            })
        
        logger.info(f"Created {len(sections)} logical sections")
        
        # Process each section with the text splitter
        for i, section in enumerate(sections):
            # Create chunks for this section
            section_chunks = self.text_splitter.create_documents(
                [section["content"]],
                metadatas=[{
                    "section": section["title"],
                    "page_number": section["page"],
                    "section_index": i,
                    "type": "text",
                    "chunking_strategy": "semantic"
                }]
            )
            
            # Add chunks to the result
            chunks.extend(section_chunks)
        
        logger.info(f"Created {len(chunks)} total chunks")
        return chunks
    
    def store_chunks(self, chunks: List[Document], collection_name: Optional[str] = None) -> None:
        """
        Store chunks in Qdrant.
        
        Args:
            chunks: List of chunks
            collection_name: Optional custom collection name
        """
        try:
            if collection_name is None:
                collection_name = SEMANTIC_COLLECTION_NAME
            
            # Convert chunks to texts and metadata
            texts = []
            metadatas = []
            
            for chunk in chunks:
                # Handle both Document objects and dictionaries
                if isinstance(chunk, Document):
                    texts.append(chunk.page_content)
                    metadatas.append(chunk.metadata)
                else:
                    texts.append(chunk["content"])
                    metadatas.append(chunk.get("metadata", {}))
            
            # Store documents in Qdrant using the correct method name
            self.storage.store_documents(
                collection_name=collection_name,
                texts=texts,
                embeddings=self.embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"Stored {len(chunks)} chunks in collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Error storing chunks in collection '{collection_name}': {e}")
            raise
    
    def process_and_store(self, elements: List[Dict], collection_name: Optional[str] = None) -> List[Document]:
        """
        Process elements and store chunks in Qdrant.
        
        Args:
            elements: List of structured elements
            collection_name: Optional custom collection name
            
        Returns:
            List of processed chunks
        """
        try:
            # Process elements into chunks
            chunks = self.process_elements(elements)
            
            # Store chunks using the correct method
            self.store_chunks(chunks, collection_name)
            
            logger.info(f"Successfully processed and stored {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in process_and_store: {e}")
            raise 