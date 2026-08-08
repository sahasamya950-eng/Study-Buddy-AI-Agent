"""
PDF processing utilities supporting pdfplumber and pypdf fallback.
"""

import io
from typing import Dict, Any, Tuple
from utils.text_cleaner import clean_text
from utils.logger import logger

def extract_text_from_pdf_stream(pdf_file_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    """
    Extracts text from PDF byte stream.
    Supports multi-page, blank page detection, password protection, and scanned PDF detection.
    
    Returns:
        Tuple of (extracted_text, metadata_dict)
    """
    extracted_pages = []
    total_pages = 0
    blank_pages = 0
    is_scanned_or_image_only = False
    
    # Try pdfplumber first
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_file_bytes)) as pdf:
            total_pages = len(pdf.pages)
            if total_pages == 0:
                return "", {"error": "PDF file contains no pages.", "pages": 0}
                
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        extracted_pages.append(f"--- Page {page_num} ---\n{text.strip()}")
                    else:
                        blank_pages += 1
                except Exception as page_err:
                    logger.warning(f"Failed to extract text from page {page_num} using pdfplumber: {page_err}")
                    blank_pages += 1

    except Exception as pdfplumber_err:
        logger.info(f"pdfplumber extraction failed/unsupported, attempting PyPDF fallback: {pdfplumber_err}")
        # Fallback to pypdf / PyPDF2
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(pdf_file_bytes))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    return "", {"error": "PDF is password protected.", "pages": 0}
                    
            total_pages = len(reader.pages)
            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        extracted_pages.append(f"--- Page {page_num} ---\n{text.strip()}")
                    else:
                        blank_pages += 1
                except Exception as page_err:
                    logger.warning(f"pypdf extraction error on page {page_num}: {page_err}")
                    blank_pages += 1
        except Exception as pypdf_err:
            logger.error(f"Both pdfplumber and pypdf failed: {pypdf_err}")
            return "", {"error": f"Failed to parse PDF file: {str(pypdf_err)}", "pages": 0}

    full_text = "\n\n".join(extracted_pages)
    cleaned_full_text = clean_text(full_text)

    # Check for unreadable / scanned PDF
    if total_pages > 0 and len(cleaned_full_text) < 20 and blank_pages >= (total_pages - 1):
        is_scanned_or_image_only = True

    metadata = {
        "total_pages": total_pages,
        "extracted_pages_count": len(extracted_pages),
        "blank_pages_count": blank_pages,
        "is_scanned_or_image": is_scanned_or_image_only,
        "character_count": len(cleaned_full_text),
        "word_count": len(cleaned_full_text.split())
    }

    if is_scanned_or_image_only:
        metadata["warning"] = "PDF appears to be scanned or contains image-only content without selectable text."

    return cleaned_full_text, metadata

def read_pdf_file_path(file_path: str) -> Tuple[str, Dict[str, Any]]:
    """Reads PDF from local file path."""
    try:
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
        return extract_text_from_pdf_stream(pdf_bytes)
    except Exception as e:
        logger.error(f"Error reading PDF file path {file_path}: {e}")
        return "", {"error": f"Cannot open PDF file path: {str(e)}"}
