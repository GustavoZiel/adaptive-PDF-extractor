"""LLM model initialization, prompts, and agent creation.

This module centralizes all LLM-related functionality:
- Model initialization (OpenAI/Gemini)
- Prompt templates for extraction and rule generation
- LangChain agent creation helpers
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from logger import get_logger

load_dotenv()

logger = get_logger(__name__)


# ============================================================================
# SECTION 1: Prompt Templates
# ============================================================================

EXTRACTION_PROMPT = r"""
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

RULE_GENERATION_PROMPT = r"""
You are an expert automation engineer specializing in **robust text extraction** from semi-structured documents.
Your task is to generate **a single, robust extraction rule** for a specific data field.

The goal is to create an **atomic rule** that reliably finds this value in future documents.
The rule MUST be based on **stable anchor keywords** (like "Inscrição", "Endereço Profissional") or patterns directly related to the field itself, **not the position of other fields**.

---
**SYSTEM-LEVEL ROBUSTNESS REQUIREMENTS**

1.  **Stable Anchors Only:** Rules must rely on stable textual anchors (labels or keywords), **never visual layout, indentation, or order**.
2.  **Cross-Field Independence:** Each rule must work independently, even if other fields are missing or reordered.
3.  **Invariant to spacing, capitalization, and accents:** Regex should be case-insensitive (`(?i)`) and tolerant to minor spacing or punctuation variations.
4.  **Anchor Proximity:** Capture text **within 1–2 lines of the anchor**, not from distant parts of the document.
5.  **No Document-Specific Values:** Avoid hardcoding specific values from the current document.

---
**CRUCIAL CONSTRAINTS: WHAT TO AVOID**

1.  **Coupled Rules**
    * Bad: "Find the text on the line after the 'inscricao' field."
    * Good: "Find the text immediately after the keyword 'Subseção'."

2.  **Keyword vs. Value Confusion**
    * Extraction Rule must capture the **value**, not the anchor.
    * Bad: `(Situação)` → matches anchor, not value.
    * Good: `Situação\s+([^\n]+)` → captures the actual value.

3.  **Permissive Validation**
    * `validation_regex` must be specific and fail on contamination from other fields.
    * Bad: `^[A-Z\s]+$` → would match unrelated field text.
    * Good: `^(?!.*(?i:{{other_keywords_joined}}))[A-ZÀ-ÖØ-öø-ÿ''\-\s]{{2,120}}$`

---
**RULE GENERATION STEPS:**

**STEP 0 (MANDATORY): Check if field is NULL**
-   Look at the `field_value` input below
-   **IF `field_value` is `null` or `None`:**
    → You MUST use `"conditional_null"` strategy
    → Skip to the **`🚨 CRITICAL: How to Configure a "conditional_null" Strategy 🚨`** section below.
-   **IF `field_value` has a value:**
    → DO NOT use `"conditional_null"`
    → Continue to steps 1-4 below

**For Non-NULL fields:**
1.  Analyze nearby text to find the **most reliable anchor** for the field.
2.  Determine the **typical format** of the expected value (letters, digits, length, accents).
3.  Ensure **cross-field contamination** is prevented using `{other_keywords}`.
4.  Generate the final JSON rule containing:
    -   `rule` (regex capturing only the value)
    -   `validation_regex` (must fail on other field keywords)

---
**INPUTS**

🔍 **FIRST: Check field_value below to determine your strategy!**
    -   If NULL/None → Use "conditional_null" strategy
    -   If has value → Use "regex" or "next_line" strategy

1.  **Full Text (`full_text`):**
{text}

2.  **Field to Analyze (`field_name`):**
"{field_name}"

3.  **⚠️ Extracted Value (`field_value`):** 👈 CHECK THIS FIRST!
"{field_value}"

4.  **Field Description (`field_description`):**
"{field_description}"

5.  **Other Field Keywords to Exclude (`other_keywords`):**
{other_keywords}

---
**OUTPUT INSTRUCTIONS**

-   Both `"rule"` and `"validation_regex"` are mandatory.
-   Extraction regex **must include a capture group `( )` for the value**, not the anchor.

**Rule Schema**
{{
    "type": "regex or keyword",
    "rule": "Python-compatible regex pattern capturing only the value (null if not applicable)",
    "keyword": "Anchor keyword (null if regex is used)",
    "strategy": "Strategy for keyword usage (e.g., 'next_line', 'multiline_until_stop', 'conditional_null')",
    "stop_keyword": "Next field anchor (null if not applicable)",
    "line_number": "Optional, use only if position is necessary",
    "validation_regex": "Regex to validate the value format, must fail on contamination."
}}

---
**EXAMPLES (For Non-NULL fields)**

1.  **Regex Rule**
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

2.  **Keyword Rule (with value)**
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

---
**🚨 CRITICAL: How to Configure a "conditional_null" Strategy 🚨**

You are in this section because **STEP 0** determined that `field_value` is `null`.
The `"conditional_null"` strategy detects when a field is genuinely empty by checking for only whitespace between two anchor keywords.

**Your Task:**
You must create a rule anchored to the field's **own label**. The executor (the Python code) will check for whitespace between this label and the *next* field's label.

**Text Example:**
```

Nome: João Silva
Seccional: OAB-SP
Categoria

Endereco Profissional: Rua ABC

```
**Analysis:** The label "Categoria" is present, but its value is empty. The *next* field is "Endereco Profissional".

**How to Configure:**
* `"type"`: `"keyword"`
* `"strategy"`: `"conditional_null"`
* `"keyword"`: The field's **own** label (e.g., `"Categoria"`). Find this label in the text.
* `"stop_keyword"`: The **next** field's label (e.g., `"Endereco Profissional"`).
* `"validation_regex"`: `"^__NULL__$"`
* `"rule"`: `null`

**Special Case: Last Field is NULL**
If the empty field is the *last* one in the document, it's the same logic, but there is no "next" field.

**Text Example:**
```

...
Telefone: +55 11 1234-5678
Situacao
(end of document)

```
**How to Configure:**
* `"keyword"`: The field's **own** label (e.g., `"Situacao"`)
* `"stop_keyword"`: `null` (This tells the system to check for whitespace until the end of the text)

**Summary: `conditional_null` Checklist**
-   **Always set these:**
    -   `"type"`: `"keyword"`
    -   `"strategy"`: `"conditional_null"`
    -   `"validation_regex"`: `"^__NULL__$"`
    -   `"rule"`: `null`
-   **Set these based on the text:**
    -   `"keyword"`: The field's **own** label (e.g., "Categoria", "Situacao").
    -   `"stop_keyword"`: The **next** field's label (or `null` if it's the last field).

*(Note: If the field's own label is not in the text, the rule should still be generated with the field's own label.*

---
**Your Turn**

Generate the extraction rule for the field `"{field_name}"`.
"""


# ============================================================================
# SECTION 2: Model Initialization
# ============================================================================


def init_model(max_retries: int = 0, timeout: int = 30):
    """Initialize LLM model based on available API keys.

    Checks for OPENAI_API_KEY and GEMINI_API_KEY environment variables
    and initializes the appropriate model.

    Args:
        max_retries: Maximum number of retries for API calls.
        timeout: Timeout for API calls in seconds.

    Returns:
        Initialized chat model instance.

    Raises:
        ValueError: If neither API key is found.
    """
    if os.getenv("OPENAI_API_KEY"):
        logger.debug("Initializing OpenAI model")
        return init_chat_model(
            "gpt-5-mini",
            model_provider="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_retries=max_retries,
            timeout=timeout,
        )
    elif os.getenv("GEMINI_API_KEY"):
        logger.debug("Initializing Gemini model")
        return init_chat_model(
            # "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            model_provider="google_genai",
            api_key=os.getenv("GEMINI_API_KEY"),
            max_retries=max_retries,
            timeout=timeout,
        )
    else:
        raise ValueError(
            "Neither GEMINI_API_KEY nor OPENAI_API_KEY is set in the environment."
        )


# ============================================================================
# SECTION 3: Agent Creation
# ============================================================================


def create_extraction_agent(model, response_format):
    """Create LangChain agent for data extraction.

    Args:
        model: Initialized LLM model.
        response_format: Pydantic model for structured output.

    Returns:
        Configured LangChain agent.
    """
    return create_agent(
        model=model,
        tools=[],
        response_format=response_format,
    )


def create_rule_agent(model, response_format):
    """Create LangChain agent for rule generation.

    Args:
        model: Initialized LLM model.
        response_format: Pydantic Rule model for structured output.

    Returns:
        Configured LangChain agent.
    """
    return create_agent(
        model=model,
        tools=[],
        response_format=response_format,
    )
