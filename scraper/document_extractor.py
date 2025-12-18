"""
PDF and image table extraction module.
Downloads and extracts text and tables from PDFs and images.
"""
import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse
import aiohttp
import asyncio

from utils.logger import setup_logger

logger = setup_logger(__name__)

# Check for optional dependencies
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logger.warning("PyPDF2 not installed. PDF extraction will be limited. Install with: pip install PyPDF2")

try:
    import pdfplumber
    PDFPLUMBER_SUPPORT = True
except ImportError:
    PDFPLUMBER_SUPPORT = False
    logger.warning("pdfplumber not installed. Table extraction from PDFs will be disabled. Install with: pip install pdfplumber")

try:
    from PIL import Image
    import pytesseract
    OCR_SUPPORT = True
except ImportError:
    OCR_SUPPORT = False
    logger.warning("PIL/pytesseract not installed. OCR for images will be disabled. Install with: pip install Pillow pytesseract")

try:
    import cv2
    import numpy as np
    CV2_SUPPORT = True
except ImportError:
    CV2_SUPPORT = False
    logger.info("opencv-python not installed. Advanced image preprocessing disabled (optional). Install with: pip install opencv-python")

try:
    import tabula
    TABULA_SUPPORT = True
    logger.info("tabula-py available for advanced PDF table extraction")
except ImportError:
    TABULA_SUPPORT = False
    logger.info("tabula-py not installed. Will use pdfplumber only for tables. Install with: pip install tabula-py (requires Java)")


class DocumentExtractor:
    """Extract text and tables from PDFs and images."""
    
    def __init__(self, download_dir: str = "downloads"):
        """
        Initialize document extractor.
        
        Args:
            download_dir: Directory to save downloaded documents
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Document downloads will be saved to: {self.download_dir}")

        # Manifest for dedupe across crawler + web search downloads.
        try:
            from utils.download_manifest import DownloadManifest

            self._manifest = DownloadManifest(self.download_dir / "manifest.json")
        except Exception:
            self._manifest = None
    
    def _sanitize_filename(self, url: str, prefix: str = "") -> str:
        """
        Create a safe filename from URL.
        
        Args:
            url: Source URL
            prefix: Optional prefix for the filename
            
        Returns:
            Safe filename
        """
        # Extract filename from URL
        parsed = urlparse(url)
        path = parsed.path
        filename = os.path.basename(path)
        
        # If no filename in URL, create one from hash
        if not filename or '.' not in filename:
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            ext = self._guess_extension(url)
            filename = f"document_{url_hash}{ext}"
        
        # Sanitize filename
        filename = re.sub(r'[^\w\s.-]', '_', filename)
        
        if prefix:
            filename = f"{prefix}_{filename}"
        
        return filename
    
    def _guess_extension(self, url: str) -> str:
        """Guess file extension from URL or content type."""
        url_lower = url.lower()
        if '.pdf' in url_lower:
            return '.pdf'
        elif '.png' in url_lower:
            return '.png'
        elif '.jpg' in url_lower or '.jpeg' in url_lower:
            return '.jpg'
        elif '.ppt' in url_lower or '.pptx' in url_lower:
            return '.pptx'
        elif '.doc' in url_lower or '.docx' in url_lower:
            return '.docx'
        return '.bin'
    
    async def download_document(
        self,
        url: str,
        query_name: Optional[str] = None,
        force: bool = False
    ) -> Optional[Path]:
        """
        Download a document from URL.
        
        Args:
            url: URL to download from
            query_name: Optional query name for organizing downloads
            force: Force download even if file exists
            
        Returns:
            Path to downloaded file or None if failed
        """
        try:
            # Always use a single flat download folder (no per-query subfolders).
            download_path = self.download_dir
            download_path.mkdir(parents=True, exist_ok=True)

            # Dedupe by URL via manifest.
            if self._manifest is not None and not force:
                existing = self._manifest.get_by_url(url)
                if existing and existing.path and Path(existing.path).exists():
                    return Path(existing.path)
            
            # Generate filename
            filename = self._sanitize_filename(url)
            filepath = download_path / filename
            
            # Skip if already downloaded
            if filepath.exists() and not force:
                logger.debug(f"Document already downloaded: {filepath}")
                if self._manifest is not None:
                    try:
                        from utils.download_manifest import sha256_file

                        self._manifest.upsert(url=url, content_hash=sha256_file(filepath), path=filepath)
                    except Exception:
                        pass
                return filepath
            
            # Download file
            logger.info(f"Downloading document from: {url}")
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status == 200:
                        content = await response.read()
                        filepath.write_bytes(content)

                        # Dedupe by content hash
                        if self._manifest is not None:
                            try:
                                from utils.download_manifest import sha256_bytes

                                digest = sha256_bytes(content)
                                existing_by_hash = self._manifest.get_by_hash(digest)
                                if existing_by_hash and existing_by_hash.path and Path(existing_by_hash.path).exists():
                                    try:
                                        filepath.unlink(missing_ok=True)
                                    except Exception:
                                        pass
                                    self._manifest.upsert(url=url, content_hash=digest, path=Path(existing_by_hash.path))
                                    return Path(existing_by_hash.path)

                                stable_name = f"{digest[:10]}_{filepath.name}"
                                stable_path = download_path / stable_name
                                if stable_path != filepath and not stable_path.exists():
                                    try:
                                        filepath.replace(stable_path)
                                        filepath = stable_path
                                    except Exception:
                                        pass

                                self._manifest.upsert(url=url, content_hash=digest, path=filepath)
                            except Exception:
                                pass

                        logger.info(f"Downloaded: {filepath} ({len(content)} bytes)")
                        return filepath
                    else:
                        logger.warning(f"Failed to download {url}: HTTP {response.status}")
                        return None
        
        except asyncio.TimeoutError:
            logger.error(f"Timeout downloading {url}")
            return None
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return None
    
    def extract_text_from_pdf(self, filepath: Path) -> Optional[str]:
        """
        Extract text from PDF file.
        
        Args:
            filepath: Path to PDF file
            
        Returns:
            Extracted text or None if failed
        """
        if not PDF_SUPPORT:
            logger.warning("PyPDF2 not available. Cannot extract PDF text.")
            return None
        
        try:
            logger.info(f"Extracting text from PDF: {filepath}")
            text_parts = []
            
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                logger.debug(f"PDF has {num_pages} pages")
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            
            full_text = "\n\n".join(text_parts)
            
            # Validate text quality
            if not self._is_quality_text(full_text):
                logger.warning(f"Extracted PDF text quality is poor from {filepath}")
                return full_text  # Still return it, but warn
            
            logger.info(f"Extracted {len(full_text)} characters from PDF (quality: OK)")
            return full_text
        
        except Exception as e:
            logger.error(f"Error extracting text from PDF {filepath}: {e}")
            return None
    
    def extract_tables_from_pdf(self, filepath: Path) -> List[List[List[str]]]:
        """
        Extract tables from PDF file using a strategy pattern.
        Tries pdfplumber first, falls back to tabula-py for complex tables.
        
        Args:
            filepath: Path to PDF file
            
        Returns:
            List of tables, where each table is a list of rows, and each row is a list of cells
        """
        if not PDFPLUMBER_SUPPORT:
            logger.warning("pdfplumber not available. Cannot extract tables from PDF.")
            return []
        
        try:
            logger.info(f"Extracting tables from PDF: {filepath}")
            all_tables = []
            
            # Strategy 1: Try pdfplumber first (faster, works for most cases)
            with pdfplumber.open(filepath) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()
                    if tables:
                        logger.debug(f"pdfplumber found {len(tables)} table(s) on page {page_num}")
                        all_tables.extend(tables)
            
            # If we found tables with pdfplumber, return them
            if all_tables:
                logger.info(f"Extracted {len(all_tables)} table(s) from PDF using pdfplumber")
                return all_tables
            
            # Strategy 2: Fall back to tabula-py for complex/multi-page tables
            if TABULA_SUPPORT:
                logger.info("No tables found with pdfplumber, trying tabula-py for complex tables...")
                try:
                    
                    # Read all tables from all pages
                    tabula_tables = tabula.io.read_pdf(
                        str(filepath),
                        pages='all',
                        multiple_tables=True,
                        lattice=True,  # Better for tables with visible borders
                        stream=True    # Better for tables without borders
                    )
                    
                    if tabula_tables:
                        # Convert DataFrame tables to list format
                        for df in tabula_tables:
                            # Ensure we have a DataFrame-like object before accessing attributes
                            if not (hasattr(df, "empty") and hasattr(df, "columns") and hasattr(df, "values")):
                                logger.debug("Skipping non-DataFrame result from tabula-py")
                                continue
                            if not df.empty: # type: ignore
                                # Convert DataFrame to list of lists
                                table_data = [list(df.columns)] + df.values.tolist() # type: ignore
                                all_tables.append(table_data)
                                logger.debug(f"tabula-py extracted table with {len(table_data)} rows")
                        
                        if all_tables:
                            logger.info(f"Extracted {len(all_tables)} table(s) from PDF using tabula-py")
                            return all_tables
                
                except Exception as e:
                    logger.warning(f"tabula-py extraction failed: {e}")
            
            # No tables found with either method
            if not all_tables:
                logger.info("No tables found in PDF with any extraction method")
            
            return all_tables
        
        except Exception as e:
            logger.error(f"Error extracting tables from PDF {filepath}: {e}")
            return []
    
    def tables_to_text(self, tables: List[List[List[str]]]) -> str:
        """
        Convert extracted tables to readable text format.
        
        Args:
            tables: List of tables from extract_tables_from_pdf
            
        Returns:
            Formatted text representation of tables
        """
        text_parts = []
        
        for table_num, table in enumerate(tables, 1):
            text_parts.append(f"\n=== Table {table_num} ===\n")
            
            for row in table:
                # Clean and join cells
                cells = [str(cell).strip() if cell else "" for cell in row]
                row_text = " | ".join(cells)
                text_parts.append(row_text)
            
            text_parts.append("")  # Empty line between tables
        
        return "\n".join(text_parts)
    
    def _preprocess_image_for_ocr(self, image_path: Path) -> Optional[Path]:
        """
        Preprocess image for better OCR results using OpenCV.
        
        Args:
            image_path: Path to original image
            
        Returns:
            Path to preprocessed image or None if preprocessing failed
        """
        if not CV2_SUPPORT:
            logger.debug("OpenCV not available, skipping image preprocessing")
            return image_path
        
        try:
            logger.debug(f"Preprocessing image for OCR: {image_path}")
            
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning(f"Failed to read image: {image_path}")
                return image_path
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding (CRITICAL for table extraction)
            # THRESH_BINARY inverts if needed, OTSU automatically finds optimal threshold
            thresh = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]
            
            # Optional: Deskew/rotation correction
            # (Could be added here if needed)
            
            # Save preprocessed image
            preprocessed_path = image_path.parent / f"preprocessed_{image_path.name}"
            cv2.imwrite(str(preprocessed_path), thresh)
            
            logger.debug(f"Preprocessed image saved: {preprocessed_path}")
            return preprocessed_path
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}, using original")
            return image_path
    
    def _is_quality_text(self, text: str, min_length: int = 100) -> bool:
        """
        Check if extracted text meets quality standards.
        
        Args:
            text: Extracted text
            min_length: Minimum text length
            
        Returns:
            True if text is of sufficient quality
        """
        if not text or len(text) < min_length:
            logger.debug(f"Text too short: {len(text) if text else 0} chars")
            return False
        
        # Check ratio of alphabetic to non-alphabetic characters
        alpha_chars = sum(c.isalpha() for c in text)
        total_chars = len(text)
        alpha_ratio = alpha_chars / total_chars if total_chars > 0 else 0
        
        if alpha_ratio < 0.5:  # At least 50% alphabetic
            logger.debug(f"Low alphabetic ratio: {alpha_ratio:.2%}")
            return False
        
        # Check for common English words (simple validation)
        common_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she'
        }
        
        words = text.lower().split()
        common_word_count = sum(1 for word in words if word.strip('.,!?()[]{}') in common_words)
        common_word_ratio = common_word_count / len(words) if words else 0
        
        if common_word_ratio < 0.05:  # At least 5% common words
            logger.debug(f"Low common word ratio: {common_word_ratio:.2%}")
            return False
        
        logger.debug(f"Text quality OK: {len(text)} chars, {alpha_ratio:.1%} alpha, {common_word_ratio:.1%} common")
        return True
    
    def extract_text_from_image(self, filepath: Path) -> Optional[str]:
        """
        Extract text from image using OCR with preprocessing.
        
        Args:
            filepath: Path to image file
            
        Returns:
            Extracted text or None if failed
        """
        if not OCR_SUPPORT:
            logger.warning("PIL/pytesseract not available. Cannot perform OCR.")
            return None
        
        try:
            logger.info(f"Performing OCR on image: {filepath}")
            
            # Preprocess image for better OCR
            preprocessed_path = self._preprocess_image_for_ocr(filepath)
            if not preprocessed_path:
                preprocessed_path = filepath
            
            # Perform OCR
            image = Image.open(preprocessed_path)
            text = pytesseract.image_to_string(image)
            
            # Clean up preprocessed file if different from original
            if preprocessed_path != filepath and preprocessed_path.exists():
                try:
                    preprocessed_path.unlink()
                except:
                    pass
            
            # Validate text quality
            if not self._is_quality_text(text):
                logger.warning(f"Extracted text quality is poor from {filepath}")
                return text  # Still return it, but warn
            
            logger.info(f"Extracted {len(text)} characters from image (quality: OK)")
            return text
        
        except Exception as e:
            logger.error(f"Error performing OCR on {filepath}: {e}")
            return None
    
    async def process_document_url(
        self,
        url: str,
        query_name: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[List[List[List[str]]]]]:
        """
        Download and extract content from document URL.
        
        Args:
            url: Document URL
            query_name: Optional query name for organizing downloads
            
        Returns:
            Tuple of (extracted_text, extracted_tables)
        """
        text, tables, _path = await self.process_document_url_with_path(url, query_name=query_name)
        return text, tables

    async def process_document_url_with_path(
        self,
        url: str,
        query_name: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[List[List[List[str]]]], Optional[Path]]:
        """Like process_document_url, but also returns the downloaded filepath."""
        filepath = await self.download_document(url, query_name)
        if not filepath:
            return None, None, None
        
        # Determine file type and extract accordingly
        ext = filepath.suffix.lower()
        
        if ext == '.pdf':
            text = self.extract_text_from_pdf(filepath)
            tables = self.extract_tables_from_pdf(filepath)
            
            # Append table text to main text
            if tables:
                table_text = self.tables_to_text(tables)
                if text:
                    text = f"{text}\n\n{table_text}"
                else:
                    text = table_text
            
            return text, tables, filepath
        
        elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
            text = self.extract_text_from_image(filepath)
            return text, None, filepath
        
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return None, None, filepath
    
    def cleanup_old_downloads(self, days_old: int = 7):
        """
        Clean up old downloaded files.
        
        Args:
            days_old: Delete files older than this many days
        """
        import time
        
        cutoff_time = time.time() - (days_old * 86400)
        deleted_count = 0
        
        for filepath in self.download_dir.rglob("*"):
            if filepath.is_file():
                if filepath.stat().st_mtime < cutoff_time:
                    try:
                        filepath.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Could not delete {filepath}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old downloaded file(s)")
