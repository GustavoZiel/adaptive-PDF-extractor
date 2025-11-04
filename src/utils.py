import json
import os
import re
from typing import Any, Dict

from pydantic import BaseModel, Field, create_model
from PyPDF2 import PdfReader

from logger import get_logger

logger = get_logger(name=__name__, level="DEBUG")


def normalize_whitespace(text: str) -> str:
    logger.debug("normalize_whitespace: input=%r", text)
    result = " ".join(text.split())
    logger.debug("normalize_whitespace: output=%r", result)
    return result


def normalize_structure(text: str) -> str:
    logger.debug(
        "normalize_structure: input length=%d", len(text) if text is not None else 0
    )

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

    # 5. Remove any leading/trailing whitespace from the whole text
    result = text.strip()
    logger.debug("normalize_structure: output length=%d", len(result))
    return result


def clean_llm_output(text: str) -> str:
    logger.debug(
        "clean_llm_output: input length=%d", len(text) if text is not None else 0
    )
    # if not text:
    #     return text
    # return normalize_whitespace(normalize_structure(text))
    result = text
    logger.debug(
        "clean_llm_output: output length=%d", len(result) if result is not None else 0
    )
    return result


def read_dataset(filename: str, data_folder: str):
    path = os.path.join(data_folder, filename)
    logger.info("read_dataset: reading %s", path)
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    logger.info(
        "read_dataset: loaded %d entries",
        len(dataset) if hasattr(dataset, "__len__") else 0,
    )
    return dataset


def get_pdf_text(file_path):
    logger.info("get_pdf_text: reading PDF %s", file_path)
    reader = PdfReader(file_path)

    page_count = len(reader.pages)
    logger.debug("get_pdf_text: page_count=%d", page_count)
    assert page_count > 0, "PDF has no pages"
    assert page_count == 1, "PDF has more than one page"

    # return normalize_structure(reader.pages[0].extract_text())
    text = reader.pages[0].extract_text()
    logger.debug(
        "get_pdf_text: extracted text length=%d", len(text) if text is not None else 0
    )
    return text


def create_pydantic_model(schema: Dict[str, Any]) -> BaseModel:
    logger.debug(
        "create_pydantic_model: creating model for schema with %d fields", len(schema)
    )
    fields = {
        key: (str | None, Field(default=None, description=value))
        for key, value in schema.items()
    }
    model = create_model("DynamicModel", **fields)
    logger.debug(
        "create_pydantic_model: model created with fields=%s", list(fields.keys())
    )
    return model


def process_dataset(dataset, data_folder):
    logger.info("process_dataset: starting processing of dataset")
    for i, data in enumerate(dataset):
        if "pdf_text" in data:
            logger.debug("process_dataset: [%d] skipping already processed item", i)
            continue

        elif "pdf_path" in data:
            logger.debug("process_dataset: [%d] reading PDF at %s", i, data["pdf_path"])

            pdf_path = os.path.join(data_folder, data["pdf_path"])
            try:
                pdf_text = get_pdf_text(pdf_path)
                data.update({"pdf_text": pdf_text})
            except Exception as e:
                logger.exception(
                    "process_dataset: [%d] failed to process %s: %s", i, pdf_path, e
                )
                raise

        # TODO see with raise something here
        if "extraction_schema" not in data:
            logger.warning(
                "process_dataset: [%d] missing extraction_schema, skipping pydantic model creation",
                i,
            )
            continue
        else:
            data.update(
                {"pydantic_model": create_pydantic_model(data["extraction_schema"])}
            )

        logger.info("process_dataset: [%d] processed successfully", i)

    logger.info("process_dataset: completed processing")
    return dataset


def write_dataset(dataset, filename, data_folder):
    # Ensure the folder exists
    os.makedirs(data_folder, exist_ok=True)

    path = os.path.join(data_folder, filename)
    logger.info("write_dataset: writing dataset to %s", path)

    with open(path, "w+", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    logger.info("write_dataset: dataset written successfully")
