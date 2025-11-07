import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from logger import get_logger

load_dotenv()

logger = get_logger(__name__)


extract_prompt_en = r"""
You are a text extraction robot. Your task is to extract information from the `Input text` according to the `Extraction Schema`.

---
**Input text:**
{text}
---
**Extraction Schema:**
{schema}
---
**CRITICAL INSTRUCTIONS:**

1. **EXTRACT VERBATIM:** Extract text *exactly as it appears* in the input. Preserve original casing and punctuation.
   - Example:
     Input: "Telefone Profissional\nSITUAÇÃO REGULAR"
     Schema: {{"situacao": "Situação do profissional"}}
     ✅ Correct: {{"situacao": "SITUAÇÃO REGULAR"}}
     ❌ Incorrect: {{"situacao": "Regular"}}

2. **USE SCHEMA DESCRIPTION:** Each field's description defines its expected format,
   If a text near an anchor (like "Categoria") does not match the description, return `None` for that field.

3. **STRICT NULL POLICY:**
   If the value is missing or invalid, assign `None`,
   Never infer or guess from nearby text.

---
**Example of "STRICT NULL POLICY":**
Input: "... Categoria Endereco Profissional ..."
Schema: {{"categoria": "Categoria, pode ser ADVOGADO, ADVOGADA, SUPLEMENTAR, ESTAGIARIO, ESTAGIARIA"}}
✅ Correct: {{"categoria": None}}
❌ Incorrect: {{"categoria": "Endereco Profissional"}}
---
"""

rule_generation_prompt_template_en = r"""
You are an expert automation engineer specializing in **robust text extraction** from semi-structured documents.
Your task is to generate **a single, robust extraction rule** for a specific data field.

The goal is to create an **atomic rule** that reliably finds this value in future documents.
The rule MUST be based on **stable anchor keywords** (like "Inscrição", "Endereço Profissional") or patterns directly related to the field itself, **not the position of other fields**.

---
**SYSTEM-LEVEL ROBUSTNESS REQUIREMENTS**

1. **Stable Anchors Only:** Rules must rely on stable textual anchors (labels or keywords), **never visual layout, indentation, or order**.
2. **Cross-Field Independence:** Each rule must work independently, even if other fields are missing or reordered.
3. **Invariant to spacing, capitalization, and accents:** Regex should be case-insensitive (`(?i)`) and tolerant to minor spacing or punctuation variations.
4. **Anchor Proximity:** Capture text **within 1–2 lines of the anchor**, not from distant parts of the document.
5. **No Document-Specific Values:** Avoid hardcoding specific values from the current document.

---
**CRUCIAL CONSTRAINTS: WHAT TO AVOID**

1. **Coupled Rules**
   * Bad: "Find the text on the line after the 'inscricao' field."
   * Good: "Find the text immediately after the keyword 'Subseção'."

2. **Keyword vs. Value Confusion**
   * Extraction Rule must capture the **value**, not the anchor.
   * Bad: `(Situação)` → matches anchor, not value.
   * Good: `Situação\s+([^\n]+)` → captures the actual value.

3. **Permissive Validation**
   * `validation_regex` must be specific and fail on contamination from other fields.
   * Bad: `^[A-Z\s]+$` → would match unrelated field text.
   * Good: `^(?!.*(?i:{{other_keywords_joined}}))[A-ZÀ-ÖØ-öø-ÿ'’\-\s]{{2,120}}$`

---
**RULE GENERATION STEPS:**

1. Analyze nearby text to find the **most reliable anchor** for the field.
2. Determine the **typical format** of the expected value (letters, digits, length, accents).
3. Ensure **cross-field contamination** is prevented using `{other_keywords}`.
4. Generate the final JSON rule containing:
   - `rule` (regex capturing only the value)
   - `validation_regex` (must fail on other field keywords)

---
**INPUTS**

1. **Full Text (`full_text`):**
{text}

2. **Field to Analyze (`field_name`):**
"{field_name}"

3. **Extracted Value (`field_value`):**
"{field_value}"

4. **Field Description (`field_description`):**
"{field_description}"

5. **Other Field Keywords to Exclude (`other_keywords`):**
{other_keywords}

---
**OUTPUT INSTRUCTIONS**

- Both `"rule"` and `"validation_regex"` are mandatory.
- Extraction regex **must include a capture group `( )` for the value**, not the anchor.

**Rule Schema**
{{
    "type": "regex or keyword",
    "rule": "Python-compatible regex pattern capturing only the value (null if not applicable)",
    "keyword": "Anchor keyword (null if regex is used)",
    "strategy": "Strategy for keyword usage (e.g., 'next_line', 'multiline_until_stop')",
    "stop_keyword": "Next field anchor (null if not applicable)",
    "line_number": "Optional, use only if position is necessary",
    "validation_regex": "Regex to validate the value format, must fail on contamination."
}}

---
**EXAMPLES**

1. **Regex Rule**
* field_name: "inscricao"
* field_value: "101943"
* other_keywords: ['nome','seccional','subsecao','categoria','situacao']
{{
    "type": "regex",
    "rule": "Inscrição[^\d]*(\d{{6}})",
    "keyword": null,
    "strategy": null,
    "stop_keyword": null,
    "line_number": null,
    "validation_regex": "^\d{{6}}$"
}}

2. **Keyword Rule (with value)**
* field_name: "subsecao"
* field_value: "Conselho Seccional - Paraná"
* other_keywords: ['nome','inscricao','seccional','categoria','situacao']
{{
    "type": "keyword",
    "rule": null,
    "keyword": "Subseção",
    "strategy": "next_line",
    "stop_keyword": null,
    "line_number": null,
    "validation_regex": "^(?!.*(?i:nome|inscricao|seccional|categoria|situacao)).*[A-ZÀ-ÖØ-öø-ÿ''\-\s]+$"
}}

3. **Conditional Null Rule (field is empty/null)**
* field_name: "categoria"
* field_value: null
* other_keywords: ['nome','inscricao','endereco','situacao']
{{
    "type": "keyword",
    "rule": null,
    "keyword": "Categoria",
    "strategy": "conditional_null",
    "stop_keyword": "Endereco Profissional",
    "line_number": null,
    "validation_regex": "^__NULL__$"
}}

**IMPORTANT: Conditional Null Strategy**

Use `"conditional_null"` strategy ONLY when the field value is null/empty/missing.

This strategy:
- Checks if there's only whitespace between `keyword` and `stop_keyword`
- Returns "__NULL__" if field is genuinely empty
- Returns None (rule fails) if field has any value
- **Works for last fields too** (when no field comes after)

Requirements:
1. `keyword`: The field label/anchor (e.g., "Categoria")
2. `stop_keyword`: OPTIONAL - Next field name OR null for last field
3. `validation_regex`: MUST be exactly `"^__NULL__$"` for null fields

**Case 1: Field with next field (stop_keyword specified)**
Example text for NULL field:
```
Categoria
Endereco Profissional Rua ABC
```
Between "Categoria" and "Endereco Profissional" there's only whitespace → NULL

**Case 2: Last field in document (stop_keyword can be null or non-existent)**
Example text for NULL "situacao" (last field):
```
Telefone Profissional +55 11 1234-5678
Situacao
```
After "Situacao" there's only whitespace until end → NULL
Use: `"stop_keyword": null` or specify next field that won't be found

Example text for NON-NULL field (don't use conditional_null):
```
Categoria
ADVOGADO
Endereco Profissional Rua ABC
```
Between keywords there's "ADVOGADO" → NOT NULL (use different strategy)
    "strategy": "next_line",
    "stop_keyword": null,
    "line_number": null,
    "validation_regex": "^(?!.*(?i:nome|inscricao|seccional|categoria|situacao)).*[A-ZÀ-ÖØ-öø-ÿ'’\-\s]+$"
}}

---
**Your Turn**

Generate the extraction rule for the field `"{field_name}"`.
"""


def init_model():
    model = None
    if os.getenv("OPENAI_API_KEY"):
        logger.debug("Initializing OpenAI model")
        model = init_chat_model(
            "gpt-5-mini",
            model_provider="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_retries=0,
            timeout=30,
        )
    elif os.getenv("GEMINI_API_KEY"):
        logger.debug("Initializing Gemini model")
        model = init_chat_model(
            "gemini-2.5-flash-lite",
            model_provider="google_genai",
            api_key=os.getenv("GEMINI_API_KEY"),
            max_retries=0,
            timeout=30,
        )
    else:
        raise ValueError(
            "Neither GEMINI_API_KEY nor OPENAI_API_KEY is set in the environment."
        )
    return model
