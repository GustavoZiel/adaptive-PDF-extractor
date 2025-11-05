import logging
import os
import time
from collections import defaultdict

import tyro
import weave
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, DirectoryPath, Field

import wandb
from cache import Cache
from logger import get_logger, set_global_log_level
from models import extract_prompt_en
from rule import Rule, generate_robust_rule
from utils import (
    create_pydantic_model,
    normalize_text,
    process_dataset,
    read_dataset,
)

load_dotenv()

logger = None
model = None


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
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    use_wandb: bool = Field(
        default=False,
        description="Enable Weights & Biases (wandb) logging for experiment tracking",
    )
    wandb_project: str = Field(
        default="adaptive-pdf-extractor",
        description="Wandb project name for logging",
    )
    wandb_run_name: str | None = Field(
        default=None,
        description="Optional custom name for the wandb run",
    )


def read_cache_file(cache_filename: str, data_folder: str) -> Cache | None:
    cache_path = os.path.join(data_folder, cache_filename)

    if not os.path.exists(cache_path):
        logger.warning("Cache file not found: %s", cache_path)
        return None

    try:
        loaded_cache = Cache.load_from_file_json(cache_path)
        if loaded_cache:
            logger.info("Loaded cache from %s", cache_path)
            return loaded_cache
        else:
            logger.warning(
                "Failed to load cache from %s (possibly empty or invalid)", cache_path
            )
            return None
    except Exception as e:
        logger.error("Error loading cache from %s: %s", cache_path, str(e))
        return None


def evaluate_performance(extracted: dict, expected: dict) -> float:
    num_fields_expected = len(expected)
    if num_fields_expected == 0:
        return 0.0
    num_correct = sum(
        1 for k, v in expected.items() if k in extracted and extracted[k] == v
    )
    pct_correct = (num_correct / num_fields_expected) * 100.0
    logger.info(
        "Performance: %d/%d fields correct (%.2f%%)",
        num_correct,
        num_fields_expected,
        pct_correct,
    )
    return pct_correct


def _define_metrics():
    wandb.define_metric("doc_index")
    wandb.define_metric("*", step_metric="doc_index")
    wandb.define_metric("accuracy", summary="mean")
    wandb.define_metric("accuracy", summary="last")
    wandb.define_metric("accuracy", summary="max")
    wandb.define_metric("accuracy", summary="min")
    wandb.define_metric("time/processing_time", summary="mean")
    wandb.define_metric("time/processing_time", summary="last")
    wandb.define_metric("time/processing_time", summary="max")
    wandb.define_metric("time/processing_time", summary="min")
    wandb.define_metric("time", summary="mean")
    wandb.define_metric("time", summary="last")
    wandb.define_metric("time", summary="max")
    wandb.define_metric("time", summary="min")


def set_wandb(args: Args):
    # Initialize wandb if enabled
    wandb_run = None
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key or wandb_api_key == "your_wandb_api_key_here":
        logger.warning(
            "WANDB_API_KEY not set in .env file. Please set it to use wandb logging."
        )
        logger.warning("Proceeding without wandb logging...")
    else:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=args.model_dump(),
        )
        weave.init(project_name=args.wandb_project)
        _define_metrics()
        logger.info(
            "Wandb and Weave initialized - Project: %s, Run: %s",
            args.wandb_project,
            wandb_run.name if wandb_run else "N/A",
        )


def init_model():
    global model

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


def main(args: Args):
    """Main processing loop: extract fields using cached rules or generate new ones."""
    logger.info("Starting extraction pipeline with args: %s", args)

    init_model()

    # Initialize wandb if enabled
    if args.use_wandb:
        wandb_dict = {}
        set_wandb(args)

    dataset = read_dataset(filename=args.dataset_filename, data_folder=args.data_folder)
    processed_dataset = process_dataset(dataset=dataset, data_folder=args.data_folder)

    dict_caches = defaultdict(Cache)

    logger.info("Initialized cache dictionary for storing rules by label")

    # Load cache once at the beginning if provided
    if args.cache_filename:
        global_loaded_cache = read_cache_file(args.cache_filename, args.data_folder)
    else:
        global_loaded_cache = None

    # Process each document in the dataset
    for doc_idx, data in enumerate(processed_dataset[:10], 1):
        start = time.time()

        logger.info("=" * 160)
        logger.info(
            f"Processing document {doc_idx}/{len(processed_dataset)} - Label: '{data['label']}'"
        )

        # Use loaded cache for this label (only on first encounter)
        if global_loaded_cache and data["label"] not in dict_caches:
            dict_caches[data["label"]] = global_loaded_cache
            logger.debug("Using loaded cache for label '%s'", data["label"])

        label_cache = dict_caches[data["label"]]
        cached_rules_count = sum(len(rules) for rules in label_cache.fields.values())

        logger.debug(
            "Label '%s' has %d cached rules across all fields",
            data["label"],
            cached_rules_count,
        )

        # Extract all required fields for this document
        all_fields = list(data["extraction_schema"].keys())
        logger.debug("Document requires %d fields: %s", len(all_fields), all_fields)

        success_fields = []
        failed_fields = []
        inserted_rules_count = 0
        text_data = data.get("pdf_text", "")

        logger.debug(f"Document text:\n[{text_data}]")

        # Attempt extraction for each field using cached rules
        ans = {}
        for field in all_fields[:1]:
            field_rules_count = len(label_cache.fields[field])
            logger.debug("Field '%s': trying %d cached rules", field, field_rules_count)

            extracted_text = label_cache.try_extract(field, text_data)

            if extracted_text is not None:
                # Extraction succeeded
                ans[field] = extracted_text
                success_fields.append(field)
                logger.info("✓ Field '%s' extracted: '%s'", field, extracted_text)
            else:
                # Extraction failed - no cached rule worked
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
        logger.debug("Successful fields: %s", success_fields)
        logger.warning("Failed fields: %s", failed_fields)
        logger.debug("Extracted values: %s", ans)

        if not failed_fields:
            logger.info("All fields extracted successfully - skipping rule generation")
            logger.info("Full answer: %s", ans)

        else:
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

            raw_response = response["structured_response"].model_dump()
            logger.debug(f"Raw LLM response: ({raw_response})")

            normalized_response = {
                k: normalize_text(v) for k, v in raw_response.items()
            }
            logger.debug(f"Normalized LLM response: ({normalized_response})")

            ans.update(normalized_response)
            logger.info("Full answer with LLM extractions: %s", ans)

            # Initialize LLM agent for rule generation
            agent_rule = create_agent(
                model=model,
                tools=[],
                response_format=Rule,
            )
            logger.debug("Initialized rule generation agent")

            # Generate extraction rules for each successfully extracted field
            llm_response = normalized_response
            generated_rules = {}

            logger.debug("Starting rule generation for %d fields", len(llm_response))
            for field, value in llm_response.items():
                # Skip None values - no rule can be generated
                if value is None:
                    logger.warning(
                        "Field '%s' has None value - skipping rule generation",
                        field,
                    )
                    continue

                logger.info(
                    "Generating rule for field '%s' with value '%s'", field, value
                )

                # Prepare inputs for rule generation
                field_description = data["extraction_schema"][field]
                logger.debug(
                    "Rule generation inputs - Field: '%s', Value: '%s', Description: '%s'",
                    field,
                    value,
                    field_description,
                )

                # Generate rule with validation loop (max_attempts retries)
                rule_object = generate_robust_rule(
                    agent_rule,
                    text_data,
                    field,
                    value,
                    field_description,
                    max_attempts=args.max_attempts,
                )

                if rule_object is not None:
                    generated_rules[field] = rule_object
                    logger.info(
                        f"✓ Generated rule for field '{field}': (type={rule_object.type}, rule={rule_object.rule or rule_object.keyword}, validation_regex={rule_object.validation_regex})",
                    )

                    # Add rule to cache immediately
                    logger.debug(
                        f"Adding rule to cache - Label: '{data['label']}', Field: '{field}'",
                    )
                    # logger.debug("Rule details: %s", rule_object.model_dump())
                    label_cache.fields[field].add_rule(rule_object)

                    if args.cache_filename:
                        # Save cache to disk immediately after adding rule
                        label_cache.save_to_file_json(
                            filename=args.cache_filename, filepath=args.data_folder
                        )
                        logger.debug(f"✓ Saved cache to: {args.cache_filename}")

                    inserted_rules_count += 1
                else:
                    logger.error(
                        "✗ Failed to generate valid rule for field '%s' after %d attempts",
                        field,
                        args.max_attempts,
                    )

            logger.info(
                f"Rule generation complete - Generated {len(generated_rules)}/{len(llm_response)} rules",
            )

        end = time.time()
        elapsed = end - start

        logger.info(
            "Completed processing document %d/%d in %.2f seconds",
            doc_idx,
            len(processed_dataset),
            elapsed,
        )

        if "expected_answer" in data:
            pct_correct = evaluate_performance(ans, data["expected_answer"])
            wandb_dict.update({"accuracy": pct_correct})
            logger.info("Document %d accuracy: %.2f%%", doc_idx, pct_correct)

        if args.use_wandb:
            logger.info("Logging wandb metrics for document %d", doc_idx)

            wandb_dict.update(
                {
                    "doc_index": doc_idx,
                    "time": {"time/processing_time": elapsed},
                    "successful_fields": len(success_fields),
                    "failed_fields": len(failed_fields),
                    # "total_fields": len(all_fields),
                    "cache": {"cache/inserted_rules_count": inserted_rules_count},
                    "cache/cached_rules_count": cached_rules_count,
                    "label": data["label"],
                }
            )
            wandb.log(wandb_dict)

    # Final summary
    logger.info("=" * 80)
    logger.info("Pipeline complete - Processed %d documents", len(processed_dataset))

    if args.cache_filename:
        logger.info("Saving all caches to disk")
        for label, cache in dict_caches.items():
            if args.cache_filename:
                cache.save_to_file_json(
                    filename=args.cache_filename, filepath=args.data_folder
                )
                logger.info(
                    "Final save - Cache for label '%s' saved to: %s",
                    label,
                    args.cache_filename,
                )
        logger.info("All caches saved successfully")

        # TODO Create a wandb table with final metrics
        # # Show final plot if enabled
        # if args.plot_metrics and fig is not None:
        #     plt.ioff()  # Turn off interactive mode
        #     logger.info("Displaying final metrics plot")
        #     print("\n" + "=" * 80)
        #     print("PIPELINE METRICS SUMMARY")
        #     print("=" * 80)
        #     if accuracies:
        #         print(f"Average Accuracy: {sum(accuracies) / len(accuracies):.2f}%")
        #         print(f"Max Accuracy: {max(accuracies):.2f}%")
        #         print(f"Min Accuracy: {min(accuracies):.2f}%")
        #     if times:
        #         print(f"Average Processing Time: {sum(times) / len(times):.2f}s")
        #         print(f"Total Processing Time: {sum(times):.2f}s")
        #         print(f"Max Processing Time: {max(times):.2f}s")
        #         print(f"Min Processing Time: {min(times):.2f}s")
        #     print("=" * 80 + "\n")
        #     plt.show()  # Keep the plot window open
        #     logger.info("Plot window closed")

    # Finish wandb run
    if args.use_wandb:
        wandb.finish()
        logger.info("Wandb run finished")


if __name__ == "__main__":
    args = tyro.cli(Args)

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    set_global_log_level(log_level)

    logger = get_logger(__name__, level=log_level)

    globals()["logger"] = logger

    main(args)
