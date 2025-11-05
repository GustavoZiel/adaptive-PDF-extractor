import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from logger import get_logger

load_dotenv()

logger = get_logger(__name__)


extract_prompt_pt = """
Você receberá um trecho de texto extraído de um documento.

Com base nesse texto, extraia **exclusivamente** as informações solicitadas e retorne o resultado em **formato JSON**, seguindo rigorosamente o esquema fornecido.

Texto de entrada:
{text}

Esquema JSON esperado:
{schema}

Instruções importantes:
- Se uma informação não estiver presente no texto, atribua o valor **null** ao campo correspondente.
- Não adicione informações que não estejam explícitas no texto.
- Retorne **apenas** o JSON, sem comentários, explicações ou texto adicional.
- Garanta que o JSON seja **válido**, **bem formatado** e **compatível com o esquema**.
"""

extract_prompt_en = """
You are a text extraction robot. Your task is to extract information from the `Input text` according to the `Extraction Schema`.

Return **only** a valid JSON object.

---
**Input text:**
{text}
---
**Extraction Schema:**
{schema}
---
**Example of Correct Extraction:**

* **Example Input Text Snippet:**
    "...
    Telefone Profissional
    SITUAÇÃO REGULAR
    ..."

* **Example Schema:**
    `{{"situacao": "Situação do profissional"}}`

* **Correct Output (Verbatim):**
    `{{"situacao": "SITUAÇÃO REGULAR"}}`

* **Incorrect Output (Interpreted/Simplified):**
    `{{"situacao": "REGULAR"}}`
---
**Important Instructions:**
- **CRITICAL:** You must extract the text *verbatim* (exactly as it appears), as shown in the CORRECT example. Do not summarize, interpret, simplify, or rephrase.
- If any information is missing from the text, assign the value **null** to the corresponding field.
- Ensure your output contains **only** the JSON and nothing else (no comments, no explanations).
"""

rule_generation_prompt_template_en = r"""
You are an expert automation engineer specializing in robust text extraction.
Your task is to generate **two** mandatory items:
1.  A **single, robust extraction rule** for a specific data field.
2.  A **mandatory `validation_regex`** to verify the format of the extracted data.

The goal is to create an "atomic" rule that can find this value in future documents. The rule MUST be based on stable "anchor" keywords (like "Inscrição", "Endereço Profissional") or patterns directly related to **itself**, not based on the position of *other* fields.

**Crucial Constraint: What to AVOID**
* **DO NOT** create rules that depend on the relative position of *other* fields.
* **Bad Rule (Coupled):** "Find the text on the line after the 'inscricao' field."
* **Good Rule (Atomic):** "Find the text on the line after the keyword 'Subseção'."

---
**ANALYSIS PATHS:**

**PATH A: If `field_value` is NOT null (e.g., "JOANA D'ARC")**
1.  **Locate:** Find the `field_value` in the `full_text`.
2.  **Find Anchor:** Analyze the text *immediately* surrounding the value to find a stable, unique keyword (like "Nome", "Inscrição", etc.).
3.  **Generate Extraction Rule:** Create the best possible extraction rule (`type`, `rule`, etc.).
4.  **Generate Validation Regex:** Analyze the `field_value` and create a `validation_regex` that matches its *format*. **This is a mandatory step.**

**PATH B: If `field_value` IS null**
1.  **Locate Anchor:** Find the "anchor" keyword for the field (e.g., "Telefone Profissional") in the `full_text`.
2.  **Find Stop-Anchor:** Analyze the text *immediately following* this anchor. Find the *next* field's anchor (e.g., "SITUAÇÃO REGULAR").
3.  **Generate Extraction Rule:** Create a `conditional_null` rule.
4.  **Generate Validation Regex:** For a `null` value, the `validation_regex` **must be `null`**.

---
**INPUTS:**

**1. Full Text (`full_text`):**
{text}

**2. Field to Analyze (`field_name`):**
"{field_name}"

**3. Extracted Value (`field_value`):** (This could be `null` or `None`)
"{field_value}"

**4. Field Description (`field_description`):**
"{field_description}"

---
**OUTPUT INSTRUCTIONS:**

Return **only** a single, valid JSON object for the generated rule, strictly adhering to the following `Rule` schema.
**Both the extraction rule and the `validation_regex` are mandatory outputs** (unless `field_value` is `null`).

**Rule Schema:**
{{
    "type": "The type of rule. Use 'regex' whenever possible.",
    "rule": "The Python-compatible regex pattern. It MUST include a capture group ( ).",
    "keyword": "The 'anchor' keyword (use if 'regex' is not possible).",
    "strategy": "The strategy for the 'keyword' (e.g., 'next_line', 'multiline_until_stop', 'conditional_null').",
    "stop_keyword": "The stopping keyword for the strategy.",
    "line_number": "The line number (use 'position' only as a last resort).",
    "validation_regex": "MANDATORY. A regex to validate the *format* of the extracted value (e.g., '^\d{{6}}$'). Must be `null` if and only if `field_value` is `null`."
}}

**Example for a `regex` rule (PATH A):**
{{
    "type": "regex",
    "rule": "Inscrição[^\d]*(\d{{6}})",
    "keyword": null,
    "strategy": null,
    "stop_keyword": null,
    "line_number": null,
    "validation_regex": "^\d{{6}}$"
}}

**Example for a `keyword` rule (PATH A):**
{{
    "type": "keyword",
    "rule": null,
    "keyword": "Subseção",
    "strategy": "next_line",
    "stop_keyword": null,
    "line_number": null,
    "validation_regex": "^[A-Z\s-]+$"
}}

**Example for a `conditional_null` rule (PATH B):**
{{
    "type": "keyword",
    "rule": null,
    "keyword": "Telefone Profissional",
    "strategy": "conditional_null",
    "stop_keyword": "SITUAÇÃO",
    "line_number": null,
    "validation_regex": null
}}

**Your Turn:**
Generate the rule for the field `"{field_name}"`.
"""


def init_model():
    model = None
    if os.getenv("OPENAI_API_KEY"):
        logger.debug("Initializing OpenAI model")
        model = init_chat_model(
            "gpt-5-mini",
            model_provider="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif os.getenv("GEMINI_API_KEY"):
        logger.debug("Initializing Gemini model")
        model = init_chat_model(
            "gemini-2.5-flash-lite",
            model_provider="google_genai",
            api_key=os.getenv("GEMINI_API_KEY"),
        )
    else:
        raise ValueError(
            "Neither GEMINI_API_KEY nor OPENAI_API_KEY is set in the environment."
        )
    return model
