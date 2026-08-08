"""
PDFReaderTool - Reads and extracts text from PDF files.
"""

from typing import Type, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from utils.pdf_utils import read_pdf_file_path, extract_text_from_pdf_stream
from utils.validators import validate_text_input
from utils.logger import logger

class PDFReaderInput(BaseModel):
    file_path: Optional[str] = Field(None, description="Absolute or relative file path to the PDF document.")

class PDFReaderTool(BaseTool):
    name: str = "pdf_reader_tool"
    description: str = (
        "Reads uploaded PDF documents, extracts clean text, ignores blank pages, "
        "and detects unreadable or scanned PDFs."
    )
    args_schema: Type[BaseModel] = PDFReaderInput

    def _run(self, file_path: Optional[str] = None) -> str:
        """Executes PDF text extraction."""
        logger.info(f"PDFReaderTool called for file path: {file_path}")
        if not file_path:
            return "Error: No PDF file path provided to PDFReaderTool."
            
        text, metadata = read_pdf_file_path(file_path)
        
        if "error" in metadata:
            return f"Error reading PDF: {metadata['error']}"

        is_valid, err = validate_text_input(text, min_chars=1)
        if not is_valid:
            return f"Warning: {err} The PDF appears to be empty or unreadable."

        summary_meta = (
            f"Successfully extracted PDF content!\n"
            f"- Total Pages: {metadata.get('total_pages')}\n"
            f"- Word Count: {metadata.get('word_count')}\n"
            f"- Character Count: {metadata.get('character_count')}\n"
        )
        if metadata.get("is_scanned_or_image"):
            summary_meta += f"- Warning: {metadata.get('warning')}\n"

        return f"{summary_meta}\nExtracted Preview:\n{text[:1000]}..."

    async def _arun(self, file_path: Optional[str] = None) -> str:
        return self._run(file_path)
