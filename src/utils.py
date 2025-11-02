import json
import os
import re
from typing import Any, Dict

from pydantic import BaseModel, Field, create_model
from PyPDF2 import PdfReader

from models import rule_generation_prompt_template_en


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def normalize_structure(text: str) -> str:
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
    return text.strip()


def clean_llm_output(text: str) -> str:
    # if not text:
    #     return text
    # return normalize_whitespace(normalize_structure(text))
    return text


def read_dataset(filename: str, data_folder: str):
    with open(os.path.join(data_folder, filename), "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset


def get_pdf_text(file_path):
    reader = PdfReader(file_path)

    assert len(reader.pages) > 0, "PDF has no pages"
    assert len(reader.pages) == 1, "PDF has more than one page"

    # return normalize_structure(reader.pages[0].extract_text())
    return reader.pages[0].extract_text()


def create_pydantic_model(schema: Dict[str, Any]) -> BaseModel:
    fields = {
        key: (str | None, Field(default=None, description=value))
        for key, value in schema.items()
    }
    model = create_model("DynamicModel", **fields)
    return model


def process_dataset(dataset, data_folder):
    for data in dataset:
        pdf_path = os.path.join(data_folder, data["pdf_path"])
        pdf_text = get_pdf_text(pdf_path)
        data.update({"pdf_text": pdf_text})
        data.update(
            {"pydantic_model": create_pydantic_model(data["extraction_schema"])}
        )
    return dataset
