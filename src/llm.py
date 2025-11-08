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

RULE_GENERATION_PROMPT_NO_OTHER_KEYWORDS = r"""
You are an expert automation engineer specializing in **robust text extraction** from semi-structured documents.
Your task is to generate **a single, robust regex extraction rule** for a specific data field.

The goal is to create an **atomic rule** that reliably finds this value in future documents.
The rule MUST be based on **stable anchor keywords** (like "Inscrição", "Endereço Profissional") or patterns directly related to the field itself.

-----

**SYSTEM-LEVEL ROBUSTNESS REQUIREMENTS**

1.  **Stable Anchors Only:** Rules must rely on stable textual anchors (labels or keywords), **never visual layout, indentation, or order**.
2.  **Cross-Field Independence:** Each rule must work independently, even if other fields are missing or reordered.
3.  **Invariant to spacing, capitalization, and accents:** Regex should be case-insensitive (`(?i)`) and tolerant to minor spacing or punctuation variations.
4.  **Anchor Proximity:** Capture text **within 1-2 lines of the anchor**, not from distant parts of the document.
5.  **No Document-Specific Values:** Avoid hardcoding specific values from the current document.

-----

**CRUCIAL CONSTRAINTS: WHAT TO AVOID**

1.  **Keyword vs. Value Confusion**
    * Extraction Rule must capture the **value**, not the anchor.
    * Bad: `(Situação)` → matches anchor, not value.
    * Good: `Situação\s+([^\n]+)` → captures the actual value.

2.  **Permissive Validation**
    * `validation_regex` must be specific and fail on contamination from other fields.
    * Bad: `^[A-Z\s]+$` → would match unrelated field text.
    * Good: `^(?!.*(?i:{{other_keywords_joined}}))[A-ZÀ-ÖØ-öø-ÿ''\-\s]{{2,120}}$`

-----

**HANDLING NULL FIELDS (Empty String Values)**

When `field_value` is an empty string `""`, the field is null/empty in the document.

**CRITICAL UNDERSTANDING:**
A null field means the field's label/keyword exists BUT has no value before the next field.
The regex must **verify the absence of content**, not just capture an empty group.

**Your Task:**
Create a regex that:
1.  Finds the field's label/keyword
2.  Skips any whitespace/newlines after it
3.  **Verifies** that what follows is either:
    * Another field's keyword (from `other_keywords`)
    * End of text `$`
4.  Captures empty string `()` to indicate null

**REGEX PATTERN STRUCTURE FOR NULL FIELDS:**
```

(?i)FieldLabel[\\s\\n]*()(?=[\\s\\n]*(?:NextKeyword1|NextKeyword2|...|$))

```

**COMPONENTS EXPLAINED:**
- `(?i)` - Case insensitive matching (optional but recommended)
- `FieldLabel` - The exact field label/keyword (e.g., "Categoria", "Inscricao")
- `[\s\n]*` - Skip any whitespace/newlines after label
- `()` - Empty capture group that returns ""
- `(?=...)` - **MANDATORY LOOKAHEAD** - verifies what comes next WITHOUT consuming it
- `[\s\n]*` - INSIDE lookahead: skip whitespace before checking next keyword
- `(?:NextKeyword1|NextKeyword2|...)` - Non-capturing group with all next field keywords
- `$` - End of text (for last field case)

**CRITICAL: The lookahead has TWO parts:**
1.  `[\s\n]*` - Skip whitespace inside the lookahead
2.  `(?:Keyword1|Keyword2|...)` - Check if next non-whitespace text is a keyword

This handles cases like: `Nome   \n\n   Inscricao` (whitespace between fields)

**WHY LOOKAHEAD IS MANDATORY:**
Without lookahead, `Categoria\s*()` would match even if there's content:
```

Bad: "Categoria\\nADVOGADO" → matches and returns "" (WRONG\!)
Good: "Categoria[\\s\\n]*()(?=[\\s\\n]*(?:Endereco|$))" → NO match because ADVOGADO follows

```

The lookahead with `[\s\n]*` inside handles whitespace-only gaps:
```

"Nome   \\n   Inscricao" → lookahead skips whitespace, finds "Inscricao" → Match ✓
"Nome123Inscricao" → lookahead finds "123", not "Inscricao" → NO match ✓

````

**EDGE CASES YOU MUST HANDLE:**

1.  **Field is in the middle of document (normal case):**
    ```
    Text: "Categoria\nEndereco Profissional: Rua ABC"
    Rule: "(?i)Categoria[\s\n]*()(?=Endereco|Telefone|...|$)"
    Result: Matches ✓ (truly null)
    ```

2.  **Field is the LAST field in document:**
    ```
    Text: "...Telefone: 123456\nCategoria"
    Rule: "(?i)Categoria[\s\n]*()(?=Endereco|Telefone|...|$)"
    Result: Matches via $ ✓ (truly null, last field)
    ```

3.  **Field label exists but HAS content (should NOT match):**
    ```
    Text: "Categoria\nADVOGADO\nEndereco: Rua ABC"
    Rule: "(?i)Categoria[\s\n]*()(?=Endereco|Telefone|...|$)"
    Result: NO match ✗ (has value "ADVOGADO", not in lookahead list)
    ```

4.  **Multiple whitespaces/newlines between null field and next:**
    ```
    Text: "Categoria\n\n\nEndereco: Rua ABC"
    Rule: "(?i)Categoria[\s\n]*()(?=[\s\n]*(?:Endereco|Telefone|...|$))"
    Result: Matches ✓ ([\s\n]* in lookahead handles multiple whitespace)
    ```

5.  **Content between field label and next field (should NOT match):**
    ```
    Text: "Nome123231Inscricao: 456789"
    Rule: "(?i)Nome[\s\n]*()(?=[\s\n]*(?:Inscricao|Categoria|...|$))"
    Result: NO match ✗ (lookahead finds "123231", not "Inscricao")
    Explanation: After "Nome", lookahead skips whitespace (none here),
    then checks if next chars match keywords. Finds "1" instead → fails.
    ```

**BUILDING THE LOOKAHEAD - CRITICAL RULES:**

1.  **Include EVERY field keyword** from `other_keywords`:
    * If other_keywords = ['nome', 'inscricao', 'endereco', 'telefone']
    * Lookahead: `(?=nome|inscricao|endereco|telefone|$)`

2.  **Use case-insensitive matching** in lookahead when possible:
    * `(?=(?i:nome|inscricao|endereco)|$)`
    * OR use `(?i)` at start of full pattern

3.  **ALWAYS include `$`** at the end for last-field case

4.  **Match the EXACT keyword form** as it appears in text:
    * If text has "Endereço Profissional", use that exact string
    * Better: use partial match like `(?=Endere|Telefone|$)`

**EXAMPLE RULE FOR NULL FIELD:**
{{
  "rule": "(?i)Categoria[\\s\\n]*()(?=[\\s\\n]*(?:Endereco|Telefone|Inscricao|Nome|$))",
  "validation_regex": "^$"
}}

**This pattern:**
- `(?i)` - Case insensitive
- `Categoria` - Finds the field label
- `[\\s\\n]*` - Skips any whitespace/newlines after label
- `()` - Captures empty string
- `(?=[\\s\\n]*(?:Endereco|Telefone|Inscricao|Nome|$))` - **VERIFIES** next field or end
  - First `[\\s\\n]*` inside lookahead skips whitespace
  - Then `(?:...)` checks if next non-whitespace is a keyword or end
- **ONLY matches when field is truly empty** (no content between label and next keyword)

**WHAT HAPPENS IN EXECUTION:**
1.  Regex finds "Categoria"
2.  Skips whitespace
3.  Lookahead checks: "Is next character start of another field keyword or end?"
4.  If YES → Match succeeds, capture group returns ""
5.  If NO → Match fails, returns None (field has a value)

-----

**INPUTS**

1.  **Full Text (`full_text`):**
    {text}

2.  **Field to Analyze (`field_name`):**
    "{field_name}"

3.  **Extracted Value (`field_value`):**
    "{field_value}"
    (Note: Empty string "" means the field is null/empty)

4.  **Field Description (`field_description`):**
    "{field_description}"

5.  **Other Field Keywords to Exclude (`other_keywords`):**
    {other_keywords}

-----

**OUTPUT INSTRUCTIONS**

  - Both `"rule"` and `"validation_regex"` are mandatory.
  - Extraction regex **must include a capture group `( )` for the value**.
  - For null fields (empty string value), use an empty capture group `()`.

**Rule Schema**
{{
  "rule": "Python-compatible regex pattern capturing only the value",
  "validation_regex": "Regex to validate the value format (use ^$ for empty strings)"
}}

-----

**EXAMPLES**

1.  **Normal Field (with value)**

  * field_name: "inscricao"
  * field_value: "101943"
  * other_keywords: ['nome','seccional','subsecao','categoria','situacao']
    {{
      "rule": "Inscrição[^\\\\d]\*(\\d{{6}})",
      "validation_regex": "^\\d{{6}}$"
    }}

2.  **Null Field (empty value)**

  * field_name: "categoria"
  * field_value: "" (empty string)
  * other_keywords: ['nome','inscricao','seccional','subsecao','situacao']
    {{
      "rule": "(?i)Categoria[\\\\s\\\\n]*()(?=[\\\\s\\\\n]*(?:Nome|Inscricao|Seccional|Subsecao|Situacao|$))",
      "validation_regex": "^$"
    }}
    Explanation:
    - `(?i)` makes matching case-insensitive
    - `Categoria` finds the field label
    - `[\\\\s\\\\n]*` skips any whitespace/newlines after label
    - `()` captures empty string
    - `(?=[\\\\s\\\\n]*(?:Nome|Inscricao|...))` lookahead with inner whitespace skip
    - Handles: "Categoria\n\nNome" ✓, "Categoria123Nome" ✗ (has content)

-----

**Your Turn**

Generate the extraction rule for the field `"{field_name}"`.

**FOR NULL FIELDS (field_value = ""):**
You MUST create a regex that:
1.  Uses case-insensitive flag: `(?i)`
2.  Finds the field label (use capitalized form as it appears in text)
3.  Skips whitespace after label: `[\\s\\n]*`
4.  Captures empty: `()`
5.  **MANDATORY LOOKAHEAD** with structure: `(?=[\\s\\n]*(?:Keyword1|Keyword2|...|$))`
    -   The `[\\s\\n]*` INSIDE lookahead is crucial
    -   It allows whitespace between label and next keyword
    -   Without it, "Nome   Inscricao" wouldn't match (has spaces)
6.  validation_regex: `^$`

**Pattern Template:**
`(?i)FieldLabel[\\s\\n]*()(?=[\\s\\n]*(?:NextField1|NextField2|NextField3|...|$))`

**CRITICAL:**
-   The lookahead has TWO parts: `[\\s\\n]*` then `(?:keywords|$)`
-   This handles whitespace-only gaps between fields
-   Must include ALL keywords from `other_keywords`
-   Detects "Nome  Inscricao" as null ✓, "Nome123Inscricao" as non-null ✗
"""

# RULE_GENERATION_PROMPT = r"""
# You are an expert automation engineer specializing in **robust text extraction** from semi-structured documents.
# Your task is to generate **a single, robust regex extraction rule** for a specific data field.

# The goal is to create an **atomic rule** that reliably finds this value in future documents.
# The rule MUST be based on **stable anchor keywords** (like "Inscrição", "Endereço Profissional") or patterns directly related to the field itself.

# -----

# **SYSTEM-LEVEL ROBUSTNESS REQUIREMENTS**

# 1.  **Stable Anchors Only:** Rules must rely on stable textual anchors (labels or keywords), **never visual layout, indentation, or order**.
# 2.  **Cross-Field Independence:** Each rule must work independently, even if other fields are missing or reordered.
# 3.  **Invariant to spacing, capitalization, and accents:** Regex should be case-insensitive (`(?i)`) and tolerant to minor spacing or punctuation variations.
# 4.  **Anchor Proximity:** Capture text **within 1-2 lines of the anchor**, not from distant parts of the document.
# 5.  **No Document-Specific Values:** Avoid hardcoding specific values from the current document.

# -----

# **CRUCIAL CONSTRAINTS: WHAT TO AVOID**

# 1.  **Keyword vs. Value Confusion**
#     * Extraction Rule must capture the **value**, not the anchor.
#     * Bad: `(Situação)` → matches anchor, not value.
#     * Good: `Situação\s+([^\n]+)` → captures the actual value.

# 2.  **Permissive Validation**
#     * `validation_regex` must be specific and fail on contamination from other fields.
#     * Bad: `^[A-Z\s]+$` → would match unrelated field text.
#     * Good: `^(?!.*(?i:{{other_keywords_joined}}))[A-ZÀ-ÖØ-öø-ÿ''\-\s]{{2,120}}$`

# -----

# **HANDLING NULL FIELDS (Empty String Values)**

# When `field_value` is an empty string `""`, the field is null/empty in the document.

# **CRITICAL UNDERSTANDING:**
# A null field means the field's label/keyword exists BUT has no value before the next field.
# The regex must **verify the absence of content**, not just capture an empty group.

# **Your Task:**
# Create a regex that:
# 1.  Finds the field's label/keyword
# 2.  Skips any whitespace/newlines after it
# 3.  **Verifies** that what follows is either:
#     * Another field's keyword (from `other_keywords`)
#     * End of text `$`
# 4.  Captures empty string `()` to indicate null

# **REGEX PATTERN STRUCTURE FOR NULL FIELDS:**
# ```

# (?i)FieldLabel[\\s\\n]*()(?=[\\s\\n]*(?:NextKeyword1|NextKeyword2|...|$))

# ```

# **COMPONENTS EXPLAINED:**
# - `(?i)` - Case insensitive matching (optional but recommended)
# - `FieldLabel` - The exact field label/keyword (e.g., "Categoria", "Inscricao")
# - `[\s\n]*` - Skip any whitespace/newlines after label
# - `()` - Empty capture group that returns ""
# - `(?=...)` - **MANDATORY LOOKAHEAD** - verifies what comes next WITHOUT consuming it
# - `[\s\n]*` - INSIDE lookahead: skip whitespace before checking next keyword
# - `(?:NextKeyword1|NextKeyword2|...)` - Non-capturing group with all next field keywords
# - `$` - End of text (for last field case)

# **CRITICAL: The lookahead has TWO parts:**
# 1.  `[\s\n]*` - Skip whitespace inside the lookahead
# 2.  `(?:Keyword1|Keyword2|...)` - Check if next non-whitespace text is a keyword

# This handles cases like: `Nome   \n\n   Inscricao` (whitespace between fields)

# **WHY LOOKAHEAD IS MANDATORY:**
# Without lookahead, `Categoria\s*()` would match even if there's content:
# ```

# Bad: "Categoria\\nADVOGADO" → matches and returns "" (WRONG\!)
# Good: "Categoria[\\s\\n]*()(?=[\\s\\n]*(?:Endereco|$))" → NO match because ADVOGADO follows

# ```

# The lookahead with `[\s\n]*` inside handles whitespace-only gaps:
# ```

# "Nome   \\n   Inscricao" → lookahead skips whitespace, finds "Inscricao" → Match ✓
# "Nome123Inscricao" → lookahead finds "123", not "Inscricao" → NO match ✓

# ````

# **EDGE CASES YOU MUST HANDLE:**

# 1.  **Field is in the middle of document (normal case):**
#     ```
#     Text: "Categoria\nEndereco Profissional: Rua ABC"
#     Rule: "(?i)Categoria[\s\n]*()(?=Endereco|Telefone|...|$)"
#     Result: Matches ✓ (truly null)
#     ```

# 2.  **Field is the LAST field in document:**
#     ```
#     Text: "...Telefone: 123456\nCategoria"
#     Rule: "(?i)Categoria[\s\n]*()(?=Endereco|Telefone|...|$)"
#     Result: Matches via $ ✓ (truly null, last field)
#     ```

# 3.  **Field label exists but HAS content (should NOT match):**
#     ```
#     Text: "Categoria\nADVOGADO\nEndereco: Rua ABC"
#     Rule: "(?i)Categoria[\s\n]*()(?=Endereco|Telefone|...|$)"
#     Result: NO match ✗ (has value "ADVOGADO", not in lookahead list)
#     ```

# 4.  **Multiple whitespaces/newlines between null field and next:**
#     ```
#     Text: "Categoria\n\n\nEndereco: Rua ABC"
#     Rule: "(?i)Categoria[\s\n]*()(?=[\s\n]*(?:Endereco|Telefone|...|$))"
#     Result: Matches ✓ ([\s\n]* in lookahead handles multiple whitespace)
#     ```

# 5.  **Content between field label and next field (should NOT match):**
#     ```
#     Text: "Nome123231Inscricao: 456789"
#     Rule: "(?i)Nome[\s\n]*()(?=[\s\n]*(?:Inscricao|Categoria|...|$))"
#     Result: NO match ✗ (lookahead finds "123231", not "Inscricao")
#     Explanation: After "Nome", lookahead skips whitespace (none here),
#     then checks if next chars match keywords. Finds "1" instead → fails.
#     ```

# **BUILDING THE LOOKAHEAD - CRITICAL RULES:**

# 1.  **Include EVERY field keyword** from `other_keywords`:
#     * If other_keywords = ['nome', 'inscricao', 'endereco', 'telefone']
#     * Lookahead: `(?=nome|inscricao|endereco|telefone|$)`

# 2.  **Use case-insensitive matching** in lookahead when possible:
#     * `(?=(?i:nome|inscricao|endereco)|$)`
#     * OR use `(?i)` at start of full pattern

# 3.  **ALWAYS include `$`** at the end for last-field case

# 4.  **Match the EXACT keyword form** as it appears in text:
#     * If text has "Endereço Profissional", use that exact string
#     * Better: use partial match like `(?=Endere|Telefone|$)`

# **EXAMPLE RULE FOR NULL FIELD:**
# {{
#   "rule": "(?i)Categoria[\\s\\n]*()(?=[\\s\\n]*(?:Endereco|Telefone|Inscricao|Nome|$))",
#   "validation_regex": "^$"
# }}

# **This pattern:**
# - `(?i)` - Case insensitive
# - `Categoria` - Finds the field label
# - `[\\s\\n]*` - Skips any whitespace/newlines after label
# - `()` - Captures empty string
# - `(?=[\\s\\n]*(?:Endereco|Telefone|Inscricao|Nome|$))` - **VERIFIES** next field or end
#   - First `[\\s\\n]*` inside lookahead skips whitespace
#   - Then `(?:...)` checks if next non-whitespace is a keyword or end
# - **ONLY matches when field is truly empty** (no content between label and next keyword)

# **WHAT HAPPENS IN EXECUTION:**
# 1.  Regex finds "Categoria"
# 2.  Skips whitespace
# 3.  Lookahead checks: "Is next character start of another field keyword or end?"
# 4.  If YES → Match succeeds, capture group returns ""
# 5.  If NO → Match fails, returns None (field has a value)

# -----

# **INPUTS**

# 1.  **Full Text (`full_text`):**
#     {text}

# 2.  **Field to Analyze (`field_name`):**
#     "{field_name}"

# 3.  **Extracted Value (`field_value`):**
#     "{field_value}"
#     (Note: Empty string "" means the field is null/empty)

# 4.  **Field Description (`field_description`):**
#     "{field_description}"

# 5.  **Other Field Keywords to Exclude (`other_keywords`):**
#     {other_keywords}

# -----

# **OUTPUT INSTRUCTIONS**

#   - Both `"rule"` and `"validation_regex"` are mandatory.
#   - Extraction regex **must include a capture group `( )` for the value**.
#   - For null fields (empty string value), use an empty capture group `()`.

# **Rule Schema**
# {{
#   "rule": "Python-compatible regex pattern capturing only the value",
#   "validation_regex": "Regex to validate the value format (use ^$ for empty strings)"
# }}

# -----

# **EXAMPLES**

# 1.  **Normal Field (with value)**

#   * field_name: "inscricao"
#   * field_value: "101943"
#   * other_keywords: ['nome','seccional','subsecao','categoria','situacao']
#     {{
#       "rule": "Inscrição[^\\\\d]\*(\\d{{6}})",
#       "validation_regex": "^\\d{{6}}$"
#     }}

# 2.  **Null Field (empty value)**

#   * field_name: "categoria"
#   * field_value: "" (empty string)
#   * other_keywords: ['nome','inscricao','seccional','subsecao','situacao']
#     {{
#       "rule": "(?i)Categoria[\\\\s\\\\n]*()(?=[\\\\s\\\\n]*(?:Nome|Inscricao|Seccional|Subsecao|Situacao|$))",
#       "validation_regex": "^$"
#     }}
#     Explanation:
#     - `(?i)` makes matching case-insensitive
#     - `Categoria` finds the field label
#     - `[\\\\s\\\\n]*` skips any whitespace/newlines after label
#     - `()` captures empty string
#     - `(?=[\\\\s\\\\n]*(?:Nome|Inscricao|...))` lookahead with inner whitespace skip
#     - Handles: "Categoria\n\nNome" ✓, "Categoria123Nome" ✗ (has content)

# -----

# **Your Turn**

# Generate the extraction rule for the field `"{field_name}"`.

# **FOR NULL FIELDS (field_value = ""):**
# You MUST create a regex that:
# 1.  Uses case-insensitive flag: `(?i)`
# 2.  Finds the field label (use capitalized form as it appears in text)
# 3.  Skips whitespace after label: `[\\s\\n]*`
# 4.  Captures empty: `()`
# 5.  **MANDATORY LOOKAHEAD** with structure: `(?=[\\s\\n]*(?:Keyword1|Keyword2|...|$))`
#     -   The `[\\s\\n]*` INSIDE lookahead is crucial
#     -   It allows whitespace between label and next keyword
#     -   Without it, "Nome   Inscricao" wouldn't match (has spaces)
# 6.  validation_regex: `^$`

# **Pattern Template:**
# `(?i)FieldLabel[\\s\\n]*()(?=[\\s\\n]*(?:NextField1|NextField2|NextField3|...|$))`

# **CRITICAL:**
# -   The lookahead has TWO parts: `[\\s\\n]*` then `(?:keywords|$)`
# -   This handles whitespace-only gaps between fields
# -   Must include ALL keywords from `other_keywords`
# -   Detects "Nome  Inscricao" as null ✓, "Nome123Inscricao" as non-null ✗
# """

RULE_GENERATION_PROMPT = r"""
You are an expert automation engineer specializing in **robust text extraction** from semi-structured documents.
Your task is to generate **a single, robust regex extraction rule** for a specific data field.

The goal is to create an **atomic rule** that reliably finds this value in future documents.
The rule MUST be based on **stable anchor keywords** (like "Inscrição", "Endereço Profissional") or patterns directly related to the field itself.

-----

**SYSTEM-LEVEL ROBUSTNESS REQUIREMENTS**

1. **Stable Anchors Only:** Rules must rely on stable textual anchors (labels or keywords), **never visual layout, indentation, or order**.
2. **Cross-Field Independence:** Each rule must work independently, even if other fields are missing or reordered.
3. **Invariant to spacing, capitalization, and accents:** Regex should be case-insensitive (`(?i)`) and tolerant to minor spacing or punctuation variations.
4. **Anchor Proximity:** Capture text **within 1-2 lines of the anchor**, not from distant parts of the document.
5. **No Document-Specific Values:** Avoid hardcoding specific values from the current document.

-----

**CRUCIAL CONSTRAINTS: WHAT TO AVOID**

1. **Keyword vs. Value Confusion**
  * Extraction Rule must capture the **value**, not the anchor.
  * Bad: `(Situação)` → matches anchor, not value.
  * Good: `Situação\s+([^\n]+)` → captures the actual value.

2.  **Format Validation**
  * `validation_regex` must be specific to the value's *format* (e.g., digits, letters, length).
  * Bad (too permissive): `.*`
  * Good (specific format): `^[A-ZÀ-ÖØ-öø-ÿ''\-\s]{{2,120}}$`

-----

**HANDLING NULL FIELDS (Empty String Values)**

When `field_value` is an empty string `""`, the field is null/empty in the document.

**CRITICAL UNDERSTANDING:**
A null field means the field's label/keyword exists BUT has no value before the next field.
The regex must **verify the absence of content**, not just capture an empty group.

**Your Task:**
Create a regex that:
1. Finds the field's label/keyword
2. Skips any whitespace/newlines after it
3. **Verifies** that what follows is either:
  * Another field's keyword (from `other_keywords`)
  * End of text `$`
4. Captures empty string `()` to indicate null

**REGEX PATTERN STRUCTURE FOR NULL FIELDS:**
```

(?i)FieldLabel[\\s\\n]*()(?=[\\s\\n]*(?:NextKeyword1|NextKeyword2|...|$))

```

**COMPONENTS EXPLAINED:**
- `(?i)` - Case insensitive matching (optional but recommended)
- `FieldLabel` - The exact field label/keyword (e.g., "Categoria", "Inscricao")
- `[\s\n]*` - Skip any whitespace/newlines after label
- `()` - Empty capture group that returns ""
- `(?=...)` - **MANDATORY LOOKAHEAD** - verifies what comes next WITHOUT consuming it
- `[\s\n]*` - INSIDE lookahead: skip whitespace before checking next keyword
- `(?:NextKeyword1|NextKeyword2|...)` - Non-capturing group with all next field keywords
- `$` - End of text (for last field case)

**CRITICAL: The lookahead has TWO parts:**
1. `[\s\n]*` - Skip whitespace inside the lookahead
2. `(?:Keyword1|Keyword2|...)` - Check if next non-whitespace text is a keyword

This handles cases like: `Nome  \n\n  Inscricao` (whitespace between fields)

**WHY LOOKAHEAD IS MANDATORY:**
Without lookahead, `Categoria\s*()` would match even if there's content:
```

Bad: "Categoria\\nADVOGADO" → matches and returns "" (WRONG\!)
Good: "Categoria[\\s\\n]*()(?=[\\s\\n]*(?:Endereco|$))" → NO match because ADVOGADO follows

```

The lookahead with `[\s\n]*` inside handles whitespace-only gaps:
```

"Nome  \\n  Inscricao" → lookahead skips whitespace, finds "Inscricao" → Match ✓
"Nome123Inscricao" → lookahead finds "123", not "Inscricao" → NO match ✓

````

**EDGE CASES YOU MUST HANDLE:**

1. **Field is in the middle of document (normal case):**
  ```
  Text: "Categoria\nEndereco Profissional: Rua ABC"
  Rule: "(?i)Categoria[\s\n]*()(?=Endereco|Telefone|...|$)"
  Result: Matches ✓ (truly null)
  ```

2. **Field is the LAST field in document:**
  ```
  Text: "...Telefone: 123456\nCategoria"
  Rule: "(?i)Categoria[\s\n]*()(?=Endereco|Telefone|...|$)"
  Result: Matches via $ ✓ (truly null, last field)
  ```

3. **Field label exists but HAS content (should NOT match):**
  ```
  Text: "Categoria\nADVOGADO\nEndereco: Rua ABC"
  Rule: "(?i)Categoria[\s\n]*()(?=Endereco|Telefone|...|$)"
  Result: NO match ✗ (has value "ADVOGADO", not in lookahead list)
  ```

4. **Multiple whitespaces/newlines between null field and next:**
  ```
  Text: "Categoria\n\n\nEndereco: Rua ABC"
  Rule: "(?i)Categoria[\s\n]*()(?=[\s\n]*(?:Endereco|Telefone|...|$))"
  Result: Matches ✓ ([\s\n]* in lookahead handles multiple whitespace)
  ```

5. **Content between field label and next field (should NOT match):**
  ```
  Text: "Nome123231Inscricao: 456789"
  Rule: "(?i)Nome[\s\n]*()(?=[\s\n]*(?:Inscricao|Categoria|...|$))"
  Result: NO match ✗ (lookahead finds "123231", not "Inscricao")
  Explanation: After "Nome", lookahead skips whitespace (none here),
  then checks if next chars match keywords. Finds "1" instead → fails.
  ```

**BUILDING THE LOOKAHEAD - CRITICAL RULES:**

1. **Include EVERY field keyword** from `other_keywords`:
  * If other_keywords = ['nome', 'inscricao', 'endereco', 'telefone']
  * Lookahead: `(?=nome|inscricao|endereco|telefone|$)`

2. **Use case-insensitive matching** in lookahead when possible:
  * `(?=(?i:nome|inscricao|endereco)|$)`
  * OR use `(?i)` at start of full pattern

3. **ALWAYS include `$`** at the end for last-field case

4. **Match the EXACT keyword form** as it appears in text:
  * If text has "Endereço Profissional", use that exact string
  * Better: use partial match like `(?=Endere|Telefone|$)`

**EXAMPLE RULE FOR NULL FIELD:**
{{
 "rule": "(?i)Categoria[\\s\\n]*()(?=[\\s\\n]*(?:Endereco|Telefone|Inscricao|Nome|$))",
 "validation_regex": "^$"
}}

**This pattern:**
- `(?i)` - Case insensitive
- `Categoria` - Finds the field label
- `[\\s\\n]*` - Skips any whitespace/newlines after label
- `()` - Captures empty string
- `(?=[\\s\\n]*(?:Endereco|Telefone|Inscricao|Nome|$))` - **VERIFIES** next field or end
 - First `[\\s\\n]*` inside lookahead skips whitespace
 - Then `(?:...)` checks if next non-whitespace is a keyword or end
- **ONLY matches when field is truly empty** (no content between label and next keyword)

**WHAT HAPPENS IN EXECUTION:**
1. Regex finds "Categoria"
2. Skips whitespace
3. Lookahead checks: "Is next character start of another field keyword or end?"
4. If YES → Match succeeds, capture group returns ""
5. If NO → Match fails, returns None (field has a value)

-----

**INPUTS**

1. **Full Text (`full_text`):**
  {text}

2. **Field to Analyze (`field_name`):**
  "{field_name}"

3. **Extracted Value (`field_value`):**
  "{field_value}"
  (Note: Empty string "" means the field is null/empty)

4. **Field Description (`field_description`):**
  "{field_description}"

5. **Other Field Keywords to Exclude (`other_keywords`):**
  {other_keywords}

-----

**OUTPUT INSTRUCTIONS**

 - Both `"rule"` and `"validation_regex"` are mandatory.
 - Extraction regex **must include a capture group `( )` for the value**.
 - For null fields (empty string value), use an empty capture group `()`.

**Rule Schema**
{{
 "rule": "Python-compatible regex pattern capturing only the value",
 "validation_regex": "Regex to validate the value format (use ^$ for empty strings)"
}}

-----

**EXAMPLES**

1. **Normal Field (with value)**

 * field_name: "inscricao"
 * field_value: "101943"
 * other_keywords: ['nome','seccional','subsecao','categoria','situacao']
  {{
   "rule": "Inscrição[^\\\\d]\*(\\d{{6}})",
   "validation_regex": "^\\d{{6}}$"
  }}

2. **Null Field (empty value)**

 * field_name: "categoria"
 * field_value: "" (empty string)
 * other_keywords: ['nome','inscricao','seccional','subsecao','situacao']
  {{
   "rule": "(?i)Categoria[\\\\s\\\\n]*()(?=[\\\\s\\\\n]*(?:Nome|Inscricao|Seccional|Subsecao|Situacao|$))",
   "validation_regex": "^$"
  }}
  Explanation:
  - `(?i)` makes matching case-insensitive
  - `Categoria` finds the field label
  - `[\\\\s\\\\n]*` skips any whitespace/newlines after label
  - `()` captures empty string
  - `(?=[\\\\s\\\\n]*(?:Nome|Inscricao|...))` lookahead with inner whitespace skip
  - Handles: "Categoria\n\nNome" ✓, "Categoria123Nome" ✗ (has content)

-----

**Your Turn**

Generate the extraction rule for the field `"{field_name}"`.

**FOR NULL FIELDS (field_value = ""):**
You MUST create a regex that:
1. Uses case-insensitive flag: `(?i)`
2. Finds the field label (use capitalized form as it appears in text)
3. Skips whitespace after label: `[\\s\\n]*`
4. Captures empty: `()`
5. **MANDATORY LOOKAHEAD** with structure: `(?=[\\s\\n]*(?:Keyword1|Keyword2|...|$))`
  -  The `[\\s\\n]*` INSIDE lookahead is crucial
  -  It allows whitespace between label and next keyword
  -  Without it, "Nome  Inscricao" wouldn't match (has spaces)
6. validation_regex: `^$`

**Pattern Template:**
`(?i)FieldLabel[\\s\\n]*()(?=[\\s\\n]*(?:NextField1|NextField2|NextField3|...|$))`

**CRITICAL:**
-  The lookahead has TWO parts: `[\\s\\n]*` then `(?:keywords|$)`
-  This handles whitespace-only gaps between fields
-  Must include ALL keywords from `other_keywords`
-  Detects "Nome Inscricao" as null ✓, "Nome123Inscricao" as non-null ✗
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
