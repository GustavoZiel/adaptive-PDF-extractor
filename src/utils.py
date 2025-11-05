import os
import re
from typing import Any, Dict

import json5
from pydantic import BaseModel, Field, create_model
from PyPDF2 import PdfReader

from logger import get_logger

logger = get_logger(name=__name__)


def normalize_text(text: str) -> str:
    """Comprehensive text normalization combining structure and whitespace normalization.

    This function:
    1. Splits conjoined letters and numbers (e.g., "Seccional101943" -> "Seccional 101943")
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


def read_dataset(filename: str, data_folder: str):
    path = os.path.join(data_folder, filename)
    logger.info("reading %s", path)
    with open(path, "r", encoding="utf-8") as f:
        dataset = json5.load(f)
    logger.info("loaded %d entries", len(dataset) if hasattr(dataset, "__len__") else 0)
    return dataset


def get_pdf_text(file_path):
    logger.info("reading PDF %s", file_path)
    reader = PdfReader(file_path)

    page_count = len(reader.pages)
    logger.debug("page_count=%d", page_count)
    assert page_count > 0, "PDF has no pages"
    assert page_count == 1, "PDF has more than one page"

    text = normalize_text(reader.pages[0].extract_text())
    return text


def create_pydantic_model(schema: Dict[str, Any]) -> BaseModel:
    # logger.debug("creating model for schema with %d fields", len(schema))
    fields = {
        key: (str | None, Field(default=None, description=value))
        for key, value in schema.items()
    }
    model = create_model("DynamicModel", **fields)
    # logger.debug("model created with fields=%s", list(fields.keys()))
    return model


def process_dataset(dataset, data_folder):
    logger.info("starting processing of dataset")
    for i, data in enumerate(dataset):
        if "pdf_text" in data:
            data["pdf_text"] = normalize_text(data["pdf_text"])
            continue

        elif "pdf_path" in data:
            pdf_path = os.path.join(data_folder, data["pdf_path"])
            try:
                pdf_text = get_pdf_text(pdf_path)
                data.update({"pdf_text": pdf_text})
            except Exception as e:
                logger.exception("failed to process %s: %s", pdf_path, e)
                raise

        if "extraction_schema" not in data:
            logger.warning(
                "missing extraction_schema, skipping pydantic model creation"
            )
            continue
        else:
            data.update(
                {"pydantic_model": create_pydantic_model(data["extraction_schema"])}
            )

        logger.info("processed item %d successfully", i)

    logger.info("completed processing")
    return dataset


def write_dataset(dataset, filename, data_folder):
    # Ensure the folder exists
    os.makedirs(data_folder, exist_ok=True)

    path = os.path.join(data_folder, filename)
    logger.info("writing dataset to %s", path)

    with open(path, "w+", encoding="utf-8") as f:
        json5.dump(dataset, f, indent=2, ensure_ascii=False)

    logger.info("dataset written successfully")
