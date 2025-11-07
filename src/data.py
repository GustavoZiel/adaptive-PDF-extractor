"""Data loading, processing, and utilities.

This module handles all data-related operations including:
- Text normalization and preprocessing
- Dataset loading and processing
- PDF text extraction
- Pydantic model creation
"""

import os
import re
import unicodedata
from typing import Any, Dict, List, Tuple

import json5
from pydantic import BaseModel, Field, create_model
from PyPDF2 import PdfReader

from logger import get_logger

logger = get_logger(name=__name__)


# ============================================================================
# SECTION 1: Text Processing
# ============================================================================


def get_json_filename(name: str) -> str:
    """Ensure the filename ends with .json extension.

    Args:
        name: Input filename
    Returns:
        Filename with .json extension
    """
    return name if name.endswith(".json") else name + ".json"


def format_dict(d: dict) -> str:
    """Format a dictionary as a pretty-printed JSON string.

    Args:
        d: Dictionary to format
    Returns:
        Pretty-printed JSON string
    """
    dict_formatted = json5.dumps(d, ensure_ascii=False, indent=2)
    return dict_formatted


def _normalize_text(
    s: str, *, return_mapping: bool = False
) -> str | Tuple[str, List[int]]:
    """Normalize text and optionally return mapping back to original indices."""
    normalized_chars: List[str] = []
    index_map: List[int] = []

    for idx, char in enumerate(s):
        lower_char = char.lower()
        decomposed = unicodedata.normalize("NFD", lower_char)
        base_chars = [c for c in decomposed if unicodedata.category(c) != "Mn"]

        if not base_chars:
            continue

        for base_char in base_chars:
            if base_char == "_" or base_char.isspace():
                if normalized_chars and normalized_chars[-1] == " ":
                    continue
                normalized_chars.append(" ")
                if return_mapping:
                    index_map.append(idx)
            elif base_char.isalnum():
                normalized_chars.append(base_char)
                if return_mapping:
                    index_map.append(idx)
            else:
                # Skip punctuation characters
                continue

    # Trim leading/trailing spaces introduced during normalization
    while normalized_chars and normalized_chars[0] == " ":
        normalized_chars.pop(0)
        if return_mapping and index_map:
            index_map.pop(0)

    while normalized_chars and normalized_chars[-1] == " ":
        normalized_chars.pop()
        if return_mapping and index_map:
            index_map.pop()

    normalized_str = "".join(normalized_chars)

    if return_mapping:
        return normalized_str, index_map
    return normalized_str


def match_text(s: str) -> str:
    """Normalize text for matching.

    Lowercases, strips accents, collapses spaces, and removes punctuation.
    """
    return _normalize_text(s)  # type: ignore[return-value]


def find_match_text(text: str, keyword: str) -> int:
    """Return the index of keyword in text using accent-insensitive matching."""
    norm_text, index_map = _normalize_text(text, return_mapping=True)
    norm_keyword = match_text(keyword)

    if not norm_text or not norm_keyword:
        return -1

    pattern = r"\b" + re.escape(norm_keyword) + r"\b"
    match = re.search(pattern, norm_text)

    if not match:
        return -1

    # Map normalized start position back to the original text index
    return index_map[match.start()]


def normalize_text(text: str) -> str:
    """Comprehensive text normalization for structure and whitespace.

    This function:
    1. Splits conjoined letters and numbers (e.g., "Seccional101943"
       -> "Seccional 101943")
    2. Splits conjoined words (e.g., "GOKUInscrição" -> "GOKU Inscrição")
    3. Collapses multiple spaces/tabs into one
    4. Collapses multiple newlines into one
    5. Collapses all whitespace (including newlines) into single spaces (final cleanup)
    6. Strips leading/trailing whitespace

    Args:
        text: Input text to normalize

    Returns:
        Normalized text string, or None if input is None
    """
    if text is None:
        return text

    # 1. Add a space between conjoined letters and numbers
    # (e.g., "Seccional101943" -> "Seccional 101943")
    text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)

    # 2. Add a space between conjoined words (e.g., "GOKUInscrição")
    # This looks for a lowercase/uppercase letter, followed by an
    # uppercase and then a lowercase (start of a new word).
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", text)

    # 3. Collapse multiple spaces/tabs *on the same line* into one
    text = re.sub(r"[ \t]+", " ", text)

    # 4. Collapse multiple consecutive newlines into a single newline
    # (e.g., "\n\n\n" -> "\n")
    text = re.sub(r"\n+", "\n", text)

    # 5. Final whitespace normalization - collapse all whitespace into single spaces
    # This includes newlines, creating a single-line normalized output
    text = " ".join(text.split())

    # 6. Strip leading/trailing whitespace
    return text.strip()


# ============================================================================
# SECTION 2: Dataset Operations
# ============================================================================


def read_dataset(filename: str, data_folder: str):
    """Read dataset from JSON file.

    Args:
        filename: Name of the JSON file
        data_folder: Folder containing the file

    Returns:
        Loaded dataset (list or dict)
    """
    filename = filename if filename.endswith(".json") else filename + ".json"
    path = os.path.join(data_folder, filename)
    logger.info("reading dataset from %s", path)

    with open(path, "r", encoding="utf-8") as f:
        dataset = json5.load(f)

    logger.info("loaded %d entries", len(dataset) if hasattr(dataset, "__len__") else 0)
    return dataset


def process_dataset(dataset, data_folder):
    """Process dataset entries by extracting PDF text and creating Pydantic models.

    This function:
    1. Normalizes existing pdf_text if present
    2. Extracts text from PDF files if pdf_path is provided
    3. Creates dynamic Pydantic models from extraction_schema

    Args:
        dataset: List of dataset entries
        data_folder: Folder containing PDF files

    Returns:
        Processed dataset with pdf_text and pydantic_model added
    """
    logger.info("starting processing of dataset")

    for i, data in enumerate(dataset):
        # If pdf_text already exists, just normalize it
        if "pdf_text" in data:
            data["pdf_text"] = normalize_text(data["pdf_text"])
            logger.debug("normalized existing pdf_text for item %d", i)

        # If pdf_path is provided, extract text from PDF
        elif "pdf_path" in data:
            pdf_path = os.path.join(data_folder, data["pdf_path"])
            try:
                pdf_text = normalize_text(get_pdf_text(pdf_path))
                data.update({"pdf_text": pdf_text})
            except Exception as e:
                logger.exception("failed to process %s: %s", pdf_path, e)
                raise

        # Create Pydantic model from extraction schema
        if "extraction_schema" not in data:
            logger.warning(
                "missing extraction_schema for item %d, skipping "
                "pydantic model creation",
                i,
            )
            continue
        else:
            data.update(
                {"pydantic_model": create_pydantic_model(data["extraction_schema"])}
            )

        logger.debug("processed item %d successfully", i)

    logger.info("completed processing of %d items", len(dataset))
    return dataset


def write_dataset(dataset, filename, data_folder):
    """Write dataset to JSON file.

    Args:
        dataset: Dataset to write
        filename: Name of the output file
        data_folder: Folder to write to
    """
    # Ensure the folder exists
    os.makedirs(data_folder, exist_ok=True)

    path = os.path.join(data_folder, filename)
    logger.info("writing dataset to %s", path)

    with open(path, "w+", encoding="utf-8") as f:
        json5.dump(dataset, f, indent=2, ensure_ascii=False)

    logger.info("dataset written successfully")


# ============================================================================
# SECTION 3: PDF Operations
# ============================================================================


def get_pdf_text(file_path):
    """Extract text from a single-page PDF file.

    Args:
        file_path: Path to the PDF file

    Returns:
        Normalized text extracted from the PDF

    Raises:
        AssertionError: If PDF has no pages or more than one page
    """
    logger.info("reading PDF %s", file_path)
    reader = PdfReader(file_path)

    page_count = len(reader.pages)
    logger.debug("page_count=%d", page_count)

    assert page_count > 0, "PDF has no pages"
    assert page_count == 1, "PDF has more than one page"

    text = reader.pages[0].extract_text()
    return text


# ============================================================================
# SECTION 4: Pydantic Utilities
# ============================================================================


def create_pydantic_model(schema: Dict[str, Any]) -> BaseModel:
    """Create a dynamic Pydantic model from an extraction schema.

    Args:
        schema: Dictionary mapping field names to descriptions

    Returns:
        Dynamically created Pydantic BaseModel class

    Example:
        >>> schema = {"name": "Person's full name", "age": "Person's age"}
        >>> model = create_pydantic_model(schema)
        >>> instance = model(name="John Doe", age="30")
    """
    fields = {
        key: (str | None, Field(default=None, description=value))
        for key, value in schema.items()
    }
    model = create_model("DynamicModel", **fields)
    logger.debug("created Pydantic model with %d fields", len(fields))
    return model
