"""Core extraction pipeline logic.

This module contains the main processing functions that orchestrate
the extraction workflow: cache-based extraction, LLM extraction,
and rule generation with feedback loops.
"""

from typing import Dict, List, Optional, Tuple

from data import normalize_text
from logger import get_logger
from rule import generate_robust_rule

logger = get_logger(__name__)


# ============================================================================
# SECTION 1: Cache-Based Extraction
# ============================================================================


def extract_with_cache(
    cache,
    label: str,
    text_data: str,
    all_fields: List[str],
) -> Tuple[Dict[str, str], List[str], List[str]]:
    """Try extracting all fields using cached rules.

    This is the "fast path" - attempting extraction without LLM calls
    by applying previously learned rules from the cache.

    Args:
        cache: Cache instance containing learned rules
        label: Document label/type for cache lookup
        text_data: Raw text to extract from
        all_fields: List of all field names to extract

    Returns:
        Tuple of (answers, success_fields, failed_fields):
        - answers: Dict mapping field names to extracted values
        - success_fields: List of fields successfully extracted
        - failed_fields: List of fields that need LLM extraction
    """
    ans = {}
    success_fields = []
    failed_fields = []

    for field_name in all_fields:
        extracted_text = cache.try_extract(field_name, text_data)

        if extracted_text is not None:
            # Cache hit - rule successfully extracted value
            if extracted_text == "__NULL__":
                ans[field_name] = None
                logger.info("✓ Field '%s' extracted: NULL (field is empty)", field_name)
            else:
                ans[field_name] = extracted_text
                logger.info("✓ Field '%s' extracted: '%s'", field_name, extracted_text)
            success_fields.append(field_name)
        else:
            # Cache miss - need LLM extraction
            failed_fields.append(field_name)
            logger.warning("✗ Field '%s' failed: no cached rule matched", field_name)

    return ans, success_fields, failed_fields


# ============================================================================
# SECTION 2: LLM-Based Extraction
# ============================================================================


def extract_with_llm(
    agent,
    text_data: str,
    extraction_schema: dict,
    failed_fields: List[str],
    extraction_prompt: str,
) -> Tuple[Dict[str, str], bool]:
    """Extract failed fields using LLM agent.

    This is the "slow path" - using LLM to extract fields that
    couldn't be extracted via cached rules.

    Args:
        agent: LangChain agent configured for extraction
        text_data: Raw text to extract from
        extraction_schema: Full schema with field descriptions
        failed_fields: List of field names that need extraction
        extraction_prompt: Prompt template for extraction

    Returns:
        Tuple of (extracted_values, success_flag):
        - extracted_values: Dictionary mapping field names to extracted values (normalized)
        - success_flag: True if extraction succeeded, False if it failed
    """
    # Invoke LLM agent with extraction prompt
    try:
        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": extraction_prompt.format(
                            text=text_data,
                            schema=extraction_schema,
                        ),
                    }
                ]
            }
        )

        # Extract and normalize response
        raw_response = response["structured_response"].model_dump()
        normalized_response = {k: normalize_text(v) for k, v in raw_response.items()}

        logger.debug(
            "LLM extraction completed for %d fields: %s",
            len(failed_fields),
            list(normalized_response.keys()),
        )

        return normalized_response, True

    except Exception as e:
        logger.error("LLM extraction failed: %s", e)
        return {field: None for field in failed_fields}, False


# ============================================================================
# SECTION 3: Rule Generation
# ============================================================================


def generate_rules_for_fields(
    agent_rule,
    extracted_fields: dict,
    text_data: str,
    extraction_schema: dict,
    all_fields: List[str],
    cache,
    label: str,
    max_attempts: int,
    save_cache_fn: Optional[callable] = None,
) -> Tuple[int, int, int]:
    """Generate and cache rules for extracted fields.

    For each field successfully extracted by the LLM, this function:
    1. Generates a rule with validation (retry loop)
    2. Adds the rule to the cache if valid
    3. Optionally saves the cache to disk

    Args:
        agent_rule: LangChain agent configured for rule generation
        extracted_fields: Dict of field names to extracted values
        text_data: Raw text the values were extracted from
        extraction_schema: Schema with field descriptions
        all_fields: List of all field names (for keyword validation)
        cache: Cache instance to store rules
        label: Document label for logging
        max_attempts: Maximum retry attempts for rule generation
        save_cache_fn: Optional callback to save cache after each rule

    Returns:
        Tuple of (rules_generated, total_prompt_tokens, total_completion_tokens)
    """
    rules_generated = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    llm2_calls = 0

    for field, value in extracted_fields.items():
        field_description = extraction_schema.get(field, "")

        logger.debug(
            "Generating rule for field '%s' with value '%s'",
            field,
            value if value else "(null)",
        )

        # Generate rule with validation loop
        rule_object, prompt_toks, completion_toks, llm_calls = generate_robust_rule(
            agent_rule,
            text_data,
            field,
            value,
            field_description,
            all_fields,
            max_attempts=max_attempts,
        )

        total_prompt_tokens += prompt_toks
        total_completion_tokens += completion_toks
        llm2_calls += llm_calls

        if rule_object is not None:
            # Successfully generated and validated rule
            cache.fields[field].add_rule(rule_object)
            rules_generated += 1

            logger.info(
                "Rule added for field '%s' (label: '%s'). Total rules for this field: %d",
                field,
                label,
                len(cache.fields[field]),
            )

            # Save cache if callback provided
            if save_cache_fn:
                try:
                    save_cache_fn()
                    logger.debug("Cache saved after adding rule for field '%s'", field)
                except Exception as e:
                    logger.warning("Failed to save cache: %s", e)
        else:
            logger.warning(
                "Failed to generate valid rule for field '%s' after %d attempts",
                field,
                max_attempts,
            )

    logger.info(
        "Rule generation complete for label '%s': %d/%d rules generated successfully",
        label,
        rules_generated,
        len(extracted_fields),
    )

    return rules_generated, total_prompt_tokens, total_completion_tokens, llm2_calls


# ============================================================================
# SECTION 4: Helper Functions
# ============================================================================


def calculate_extraction_cost(
    prompt_tokens: int, completion_tokens: int, price_in: float, price_out: float
) -> float:
    """Calculate cost of LLM extraction.

    Args:
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        price_in: Cost per input token
        price_out: Cost per output token

    Returns:
        Total cost in dollars
    """
    return (prompt_tokens * price_in) + (completion_tokens * price_out)
