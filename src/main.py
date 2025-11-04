import os
from collections import defaultdict

import tyro
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, DirectoryPath, Field, FilePath

from cache import Cache
from logger import get_logger
from models import extract_prompt_en
from rule import Rule, create_empty_rule, generate_robust_rule
from utils import (
    clean_llm_output,
    create_pydantic_model,
    process_dataset,
    read_dataset,
    write_dataset,
)

load_dotenv()

logger = get_logger(__name__, level="INFO")


class Args(BaseModel):
    data_folder: DirectoryPath = Field(
        # default=DirectoryPath("data"),
        description="Path to the data folder."
    )
    dataset_filename: str = Field(
        # default="dataset.json",
        description="Name of the dataset JSON file within the data folder.",
    )
    max_attempts: int = Field(
        # default=5,
        ge=1,
        le=10,
        description="Maximum number of attempts.",
    )
    cache_filename: str | None = Field(
        None,
        description="Name of the cache file within the data folder.",
    )
    # timeout: float = Field(
    #     # default=30.0,
    #     ge=0.0,
    #     description="Timeout duration in seconds.",
    # )


model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    api_key=os.getenv("GEMINI_API_KEY"),
)


def read_cache_file(filepath: str) -> Cache | None:
    """Read cache from a JSON file if it exists."""
    if os.path.exists(filepath):
        logger.info("Loading cache from file: %s", filepath)
        return Cache.load_from_file_json(filepath)
    else:
        logger.warning("Cache file not found: %s", filepath)
        return None


def main(args: Args):
    """Main processing loop: extract fields using cached rules or generate new ones."""
    logger.info("Starting extraction pipeline with args: %s", args)

    # Load and preprocess dataset
    dataset = read_dataset(filename=args.dataset_filename, data_folder=args.data_folder)
    logger.info("Loaded %d documents from dataset", len(dataset))

    processed_dataset = process_dataset(dataset=dataset, data_folder=args.data_folder)
    logger.debug("Processed %d documents (added PDF text)", len(processed_dataset))

    dict_caches = defaultdict(Cache)

    logger.debug("Initialized cache dictionary for storing rules by label")

    # Process each document in the dataset
    for doc_idx, data in enumerate(processed_dataset, 1):
        logger.info("=" * 80)
        logger.info(
            "Processing document %d/%d - Label: '%s'",
            doc_idx,
            len(processed_dataset),
            data["label"],
        )

        # Get cache for this document's label
        if args.cache_filename:
            cache_path = os.path.join(args.data_folder, args.cache_filename)
            loaded_cache = read_cache_file(cache_path)
            if loaded_cache:
                dict_caches[data["label"]] = loaded_cache

        label_cache = dict_caches[data["label"]]
        cached_rules_count = sum(len(rules) for rules in label_cache.fields.values())
        logger.debug(
            "Label '%s' has %d cached rules across all fields",
            data["label"],
            cached_rules_count,
        )

        # Extract all required fields for this document
        all_fields = list(data["extraction_schema"].keys())
        logger.info("Document requires %d fields: %s", len(all_fields), all_fields)

        success_fields = []
        failed_fields = []
        text_data = data.get("pdf_text", "")
        logger.debug("Document text length: %d characters", len(text_data))

        # Attempt extraction for each field using cached rules
        ans = {}
        for field in all_fields:
            field_rules_count = len(label_cache.fields[field])
            logger.debug("Field '%s': trying %d cached rules", field, field_rules_count)

            extracted_text, rule_matched = label_cache.try_extract(field, text_data)
            print(
                f"Field: {field}, Extracted Text: {extracted_text}, Matched: {rule_matched}"
            )

            if rule_matched:
                # A rule successfully validated (could be regular rule or empty rule)
                ans[field] = extracted_text
                success_fields.append(field)
                if extracted_text is None:
                    logger.info("✓ Field '%s' extracted (empty): None", field)
                else:
                    logger.info("✓ Field '%s' extracted: '%s'", field, extracted_text)
            else:
                # No cached rule matched
                failed_fields.append(field)
                logger.warning("✗ Field '%s' failed: no cached rule matched", field)

        # Summary of extraction results
        logger.info(
            "Extraction summary - Success: %d/%d, Failed: %d/%d",
            len(success_fields),
            len(all_fields),
            len(failed_fields),
            len(all_fields),
        )
        logger.info("Successful fields: %s", success_fields)
        logger.warning("Failed fields: %s", failed_fields)
        logger.debug("Extracted values: %s", ans)

        logger.info("Ans so far: %s", ans)

        # Skip rule generation if all fields were successfully extracted
        if not failed_fields:
            logger.info("All fields extracted successfully - skipping rule generation")
            continue

        # Generate LLM extraction for failed fields
        logger.info("Generating new rules for %d failed fields", len(failed_fields))

        failed_fields_dict = {
            field: data["extraction_schema"][field] for field in failed_fields
        }
        logger.debug("Failed fields schema: %s", failed_fields_dict)

        # Create dynamic Pydantic model for failed fields
        data.update({"pydantic_model": create_pydantic_model(failed_fields_dict)})
        logger.debug("Created Pydantic model: %s", data["pydantic_model"].__name__)

        # Initialize LLM agent for extraction
        agent = create_agent(
            model=model,
            tools=[],
            response_format=data["pydantic_model"],
        )
        logger.debug("Initialized extraction agent with structured output")

        # Invoke LLM to extract failed fields
        logger.info("Invoking LLM to extract %d failed fields", len(failed_fields))
        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": extract_prompt_en.format(
                            text=text_data,
                            schema=data["pydantic_model"],
                        ),
                    }
                ]
            },
        )

        logger.debug("Raw LLM response: %s", response["structured_response"])

        # Normalize LLM output (clean whitespace, format values)
        model_dict = response["structured_response"].model_dump()
        normalized_dict = {k: clean_llm_output(v) for k, v in model_dict.items()}
        ans.update(normalized_dict)

        response["structured_response"] = data["pydantic_model"](**normalized_dict)

        logger.info("Normalized LLM extraction: %s", response["structured_response"])

        # Initialize LLM agent for rule generation
        agent_rule = create_agent(
            model=model,
            tools=[],
            response_format=Rule,
        )
        logger.debug("Initialized rule generation agent")

        # Generate extraction rules for each successfully extracted field
        llm_response = response["structured_response"].model_dump()
        generated_rules = {}

        logger.info("Starting rule generation for %d fields", len(llm_response))
        for field, value in llm_response.items():
            # Handle None values with special empty rule
            if value is None:
                logger.warning(
                    "Field '%s' has None value - creating empty rule to prevent LLM hallucination",
                    field,
                )
                empty_rule = create_empty_rule()
                generated_rules[field] = empty_rule
                logger.info("✓ Created empty rule for field '%s' (type=empty)", field)
                continue

            logger.info("Generating rule for field '%s' with value '%s'", field, value)

            # Prepare inputs for rule generation
            field_name = field
            field_value = value
            field_description = data["extraction_schema"][field]

            logger.debug(
                "Rule generation inputs - Field: '%s', Value: '%s', Description: '%s'",
                field_name,
                field_value,
                field_description,
            )

            # Generate rule with validation loop (max_attempts retries)
            rule_object = generate_robust_rule(
                agent_rule,
                text_data,
                field_name,
                field_value,
                field_description,
                max_attempts=args.max_attempts,
            )

            if rule_object is not None:
                generated_rules[field] = rule_object
                logger.info(
                    "✓ Generated rule for field '%s': type=%s, rule=%s",
                    field,
                    rule_object.type,
                    rule_object.rule or rule_object.keyword,
                )
            else:
                logger.error(
                    "✗ Failed to generate valid rule for field '%s' after %d attempts",
                    field,
                    args.max_attempts,
                )

        logger.info(
            "Rule generation complete - Generated %d/%d rules",
            len(generated_rules),
            len(llm_response),
        )
        logger.debug(
            "Generated rules details: %s",
            {k: v.model_dump() if v else None for k, v in generated_rules.items()},
        )

        # Add generated rules to cache
        rules_added = 0
        for field, rule in generated_rules.items():
            if rule is not None:
                logger.info(
                    "Adding rule to cache - Label: '%s', Field: '%s'",
                    data["label"],
                    field,
                )
                logger.debug("Rule details: %s", rule.model_dump())
                label_cache.fields[field].add_rule(rule)
                rules_added += 1
            else:
                logger.warning("Skipping None rule for field '%s'", field)

        logger.info(
            "Added %d new rules to cache for label '%s'", rules_added, data["label"]
        )

    # Save all caches to disk
    logger.info("=" * 80)
    logger.info("Saving caches for %d labels", len(dict_caches))

    for label, label_cache in dict_caches.items():
        total_rules = sum(len(rules) for rules in label_cache.fields.values())
        cache_filename = f"cache_{label}.json"

        logger.info("Saving cache for label '%s' - Total rules: %d", label, total_rules)
        logger.debug("Cache fields: %s", list(label_cache.fields.keys()))

        label_cache.save_to_file_json(
            filename=f"cache/{cache_filename}", filepath=args.data_folder
        )
        logger.info("✓ Saved cache to: %s", cache_filename)

    logger.info("Pipeline complete - All caches saved successfully")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
