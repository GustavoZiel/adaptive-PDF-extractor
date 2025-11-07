import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

import tyro
import weave
from cache import Cache
from dotenv import load_dotenv
from langchain.agents import create_agent
from models import extract_prompt_en, init_model
from pydantic import BaseModel, DirectoryPath, Field
from rule import Rule, generate_robust_rule

import wandb
from data import (
    create_pydantic_model,
    normalize_text,
    process_dataset,
    read_dataset,
)
from logger import get_logger, set_global_log_level

load_dotenv()

logger = None

# source: https://platform.openai.com/docs/models/gpt-5-mini
GPT_5_MINI_INPUT_TOKEN_COST = 0.25 / 1_000_000  # $0.25 per 1M input tokens
GPT_5_MINI_OUTPUT_TOKEN_COST = 2.00 / 1_000_000  # $2.00 per 1M output tokens


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
    save_ans_disk: bool = Field(
        default=True,
        description="Save final answers to disk as JSON file",
    )
    save_ans_wandb: bool = Field(
        default=True,
        description="Upload final answers to wandb as run file",
    )
    save_cache_disk: bool = Field(
        default=True,
        description="Save cache to disk as JSON file",
    )
    save_cache_wandb: bool = Field(
        default=True,
        description="Upload cache to wandb as run file",
    )


def read_cache_file(cache_filename: str, data_folder: str) -> Cache | None:
    cache_path = os.path.join(data_folder, cache_filename)

    if not os.path.exists(cache_path):
        logger.warning(f"Cache file not found: {cache_path}, skipping load.")
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
    logger.debug("Expected: %s", expected)
    logger.debug("Extracted: %s", extracted)
    logger.info(
        "Performance: %d/%d fields correct (%.2f%%)",
        num_correct,
        num_fields_expected,
        pct_correct,
    )
    return pct_correct


def _define_metrics():
    """Define all wandb metrics with proper step and summary aggregations."""
    # Define doc_index as the primary x-axis
    wandb.define_metric("doc_index")
    wandb.define_metric("*", step_metric="doc_index")

    # 1. Performance Metrics
    wandb.define_metric("performance_per_doc", summary="mean")
    wandb.define_metric("performance_per_doc", summary="max")
    wandb.define_metric("performance_per_doc", summary="min")
    wandb.define_metric("performance_per_doc", summary="last")
    wandb.define_metric("avg_performance", summary="last")

    # # 2. Cost & Efficiency Metrics
    wandb.define_metric("processing_time_sec", summary="mean")
    wandb.define_metric("processing_time_sec", summary="max")
    wandb.define_metric("processing_time_sec", summary="min")

    wandb.define_metric("avg_processing_time_sec", summary="last")

    wandb.define_metric("cost", summary="mean")
    wandb.define_metric("total_cost", summary="last")
    wandb.define_metric("avg_cost_per_doc", summary="last")

    wandb.define_metric("total_prompt_tokens", summary="last")
    wandb.define_metric("total_completion_tokens", summary="last")

    # 3. Cache & Healing Metrics
    wandb.define_metric("fast_path_success", summary="mean")
    wandb.define_metric("cache_hit_rate", summary="last")

    wandb.define_metric("total_rules_in_local_cache", summary="last")
    wandb.define_metric("total_rules_in_global_cache", summary="last")


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


@dataclass
class WandbLogProperties:
    # 🔹 Persistent (cumulative) metrics
    total_docs: int = 0
    total_cost: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    avg_cost_per_doc: float = 0.0
    avg_processing_time_sec: float = 0.0
    avg_performance: float = 0.0
    cache_hit_rate: float = 0.0
    total_rules_in_global_cache: int = 0
    total_elapsed: float = 0.0  # Total cumulative time

    # Running sums for averages
    _sum_processing_time: float = 0.0
    _sum_performance: float = 0.0
    _sum_cache_hits: int = 0

    # 🔹 Per-document (reset each iteration)
    doc_index: int = 0
    doc_id: str = ""
    label: str = ""
    processing_time_sec: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0

    # Performance metrics
    fields_correct_count: int = 0
    fields_failed_count: int = 0
    total_fields: int = 0
    performance_per_doc: float = 0.0
    failed_field_names: List[str] = field(default_factory=list)

    # Cache & Healing metrics
    fast_path_success: int = 0  # 1 if cache extracted 100%, 0 otherwise
    new_rules_added: int = 0
    total_rules_in_local_cache: int = 0

    # LLM call tracking
    llm1_extractor_calls: int = 0
    llm2_generator_calls: int = 0

    def update_per_doc(
        self,
        doc_index: int,
        doc_id: str,
        label: str,
        prompt_toks: int,
        completion_toks: int,
        price_in: float,
        price_out: float,
        processing_time: float,
        accuracy_pct: float,
        fields_correct: int,
        fields_failed: int,
        total_fields: int,
        failed_field_names: List[str],
        fast_path_success: bool,
        new_rules_added: int,
        total_rules_in_local_cache: int,
        total_rules_in_global_cache: int,
        llm1_calls: int,
        llm2_calls: int,
    ):
        """Update all per-document stats and roll up to totals."""
        self.doc_index = doc_index
        self.doc_id = doc_id
        # TODO Ver isso aqui por que é variável
        self.label = label
        self.prompt_tokens = prompt_toks
        self.completion_tokens = completion_toks
        self.processing_time_sec = processing_time

        # Performance metrics
        self.fields_correct_count = fields_correct
        self.fields_failed_count = fields_failed
        self.total_fields = total_fields
        self.performance_per_doc = accuracy_pct
        self.failed_field_names = failed_field_names

        # Cache & Healing metrics
        self.fast_path_success = 1 if fast_path_success else 0
        self.new_rules_added = new_rules_added
        self.total_rules_in_local_cache = total_rules_in_local_cache
        self.total_rules_in_global_cache = total_rules_in_global_cache

        # LLM call tracking
        self.llm1_extractor_calls = llm1_calls
        self.llm2_generator_calls = llm2_calls

        # Compute cost for this doc
        self.cost = (prompt_toks * price_in) + (completion_toks * price_out)

        # Update global aggregates
        self.total_docs += 1
        self.total_cost += self.cost
        self.total_prompt_tokens += prompt_toks
        self.total_completion_tokens += completion_toks
        self.avg_cost_per_doc = self.total_cost / self.total_docs

        # Update running averages
        self._sum_processing_time += processing_time
        self.avg_processing_time_sec = self._sum_processing_time / self.total_docs
        self.total_elapsed = self._sum_processing_time  # Total cumulative time

        self._sum_performance += self.performance_per_doc
        self.avg_performance = self._sum_performance / self.total_docs

        self._sum_cache_hits += self.fast_path_success
        self.cache_hit_rate = (self._sum_cache_hits / self.total_docs) * 100.0

    def reset_per_doc(self):
        """Reset transient (per-document) metrics."""
        self.doc_index = 0
        self.doc_id = ""
        self.label = ""
        self.processing_time_sec = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost = 0.0

        self.fields_correct_count = 0
        self.fields_failed_count = 0
        self.total_fields = 0
        self.performance_per_doc = 0.0
        self.failed_field_names.clear()

        self.fast_path_success = 0
        self.new_rules_added = 0
        self.total_rules_in_local_cache = 0

        self.llm1_extractor_calls = 0
        self.llm2_generator_calls = 0

    def to_dict(self):
        """Return a dict for wandb logging, excluding internal fields."""
        data = {
            # Document identification
            "doc_index": self.doc_index,
            # "doc_id": self.doc_id,
            "label": self.label,
            # Performance metrics
            "performance": {
                "performance/per_doc": self.performance_per_doc,
                "performance/avg": self.avg_performance,
                "performance/fields_correct": self.fields_correct_count,
                "performance/fields_failed": self.fields_failed_count,
                # performance/ "total_fields": self.total_fields,
                "performance/failed_field_names": self.failed_field_names,
            },
            # Time metrics
            "time": {
                "time/processing_time_sec": self.processing_time_sec,
                "time/avg_processing_time_sec": self.avg_processing_time_sec,
                "time/total_elapsed": self.total_elapsed,
            },
            # Cost metrics
            "cost": {
                "cost/per_doc": self.cost,
                "cost/total": self.total_cost,
                "cost/avg_per_doc": self.avg_cost_per_doc,
            },
            # Token usage
            "tokens": {
                "tokens/prompt": self.prompt_tokens,
                "tokens/completion": self.completion_tokens,
                "tokens/total_prompt": self.total_prompt_tokens,
                "tokens/total_completion": self.total_completion_tokens,
            },
            # Cache & Healing metrics
            "cache": {
                "cache/hit_rate": self.cache_hit_rate,
                "cache/fast_path_success": self.fast_path_success,
                "cache/fast_path_total_success": self._sum_cache_hits,
                "cache/new_rules_added": self.new_rules_added,
                "cache/total_rules_in_local_cache": self.total_rules_in_local_cache,
                "cache/total_rules_in_global_cache": self.total_rules_in_global_cache,
            },
            # LLM calls tracking
            "llm_calls": {
                "llm_calls/extractor": self.llm1_extractor_calls,
                "llm_calls/generator": self.llm2_generator_calls,
                "llm_calls/total": self.llm1_extractor_calls
                + self.llm2_generator_calls,
            },
        }

        return data


def main(args: Args):
    """Main processing loop: extract fields using cached rules or generate new ones."""
    logger.info("Starting extraction pipeline with args: %s", args)

    model = init_model()

    # Initialize wandb if enabled
    wandb_logger = None
    if args.use_wandb:
        wandb_logger = WandbLogProperties()
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

    # # Define which document indices to process
    # # selected_indices = [0, 74, 211, 1114, 1182, 1664]
    # selected_indices = [0]

    # # Filter processed_dataset to only include selected indices
    # filtered_dataset = [
    #     (idx, data)
    #     for idx, data in enumerate(processed_dataset)
    #     if idx in selected_indices
    # ]

    # logger.info(
    #     "Processing %d selected documents out of %d total documents",
    #     len(filtered_dataset),
    #     len(processed_dataset),
    # )

    # Collect all answers for final save
    all_answers = []

    # Process each document in the dataset
    for doc_idx, data in enumerate(processed_dataset, 1):
        start = time.time()

        success_fields = []
        failed_fields = []
        text_data = data.get("pdf_text", "")
        new_rules_added = 0

        # Token tracking for this document
        doc_prompt_tokens = 0
        doc_completion_tokens = 0
        llm1_calls = 0  # Extractor LLM calls
        llm2_calls = 0  # Rule generator LLM calls

        logger.info("=" * 120)
        logger.info(
            f"Processing document {doc_idx}/{len(processed_dataset)} - Label: '{data['label']}'"
        )

        # Use loaded cache for this label (only on first encounter)
        if global_loaded_cache and data["label"] not in dict_caches:
            dict_caches[data["label"]] = global_loaded_cache
            logger.debug("Using loaded cache for label '%s'", data["label"])

        label_cache = dict_caches[data["label"]]
        total_rules_in_local_cache = sum(
            len(rules) for rules in label_cache.fields.values()
        )

        logger.debug(
            "Label '%s' has %d cached rules across all fields",
            data["label"],
            total_rules_in_local_cache,
        )

        # Extract all required fields for this document
        all_fields = list(data["extraction_schema"].keys())
        logger.debug("Document requires %d fields: %s", len(all_fields), all_fields)

        logger.debug(f"Document text:\n[{text_data}]")

        # Attempt extraction for each field using cached rules
        ans = {}
        # for field_name in [f for f in all_fields if f == "categoria"]:
        for field_name in all_fields:
            field_rules_count = len(label_cache.fields[field_name])
            logger.debug(
                "Field '%s': trying %d cached rules", field_name, field_rules_count
            )

            extracted_text = label_cache.try_extract(field_name, text_data)

            if extracted_text is not None:
                # Extraction succeeded
                # Convert __NULL__ marker to None for final output
                if extracted_text == "__NULL__":
                    ans[field_name] = None
                    logger.info(
                        "✓ Field '%s' extracted: NULL (field is empty)", field_name
                    )
                else:
                    ans[field_name] = extracted_text
                    logger.info(
                        "✓ Field '%s' extracted: '%s'", field_name, extracted_text
                    )
                success_fields.append(field_name)
            else:
                # Extraction failed - no cached rule worked
                failed_fields.append(field_name)
                logger.warning(
                    "✗ Field '%s' failed: no cached rule matched", field_name
                )

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
            logger.debug(
                "Created Pydantic model: %s, model_dump: %s",
                data["pydantic_model"].__name__,
                data["pydantic_model"]().model_dump(),
            )

            # Initialize LLM agent for extraction
            agent = create_agent(
                model=model,
                tools=[],
                response_format=data["pydantic_model"],
            )
            logger.debug("Initialized extraction agent with structured output")

            # Invoke LLM to extract failed fields
            logger.info("Invoking LLM to extract %d failed fields", len(failed_fields))
            try:
                response = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": extract_prompt_en.format(
                                    text=text_data,
                                    # Providing only the failed fields
                                    # schema={
                                    #     k: v
                                    #     for k, v in data["extraction_schema"].items()
                                    #     if k in failed_fields
                                    # },
                                    # Providing full schema for better context
                                    schema=data["extraction_schema"],
                                ),
                            }
                        ]
                    },
                )
            except Exception as e:
                logger.error(
                    "Error during LLM extraction for document %s: %s", doc_idx, str(e)
                )
                continue

            # Track LLM1 usage (extractor)
            llm1_calls += 1
            ai_message = response["messages"][-1]
            if (
                hasattr(ai_message, "response_metadata")
                and ai_message.response_metadata
            ):
                if (
                    "token_usage" in ai_message.response_metadata
                    and ai_message.response_metadata["token_usage"]
                ):
                    doc_prompt_tokens += ai_message.response_metadata[
                        "token_usage"
                    ].get("prompt_tokens", 0)
                    doc_completion_tokens += ai_message.response_metadata[
                        "token_usage"
                    ].get("completion_tokens", 0)
                    logger.debug(
                        "LLM token usage - Prompt: %d, Completion: %d",
                        doc_prompt_tokens,
                        doc_completion_tokens,
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
                # For None values, we need to generate conditional_null rules
                # Don't skip them - they are important for detecting empty fields!
                if value is None:
                    logger.info(
                        "Field '%s' has None value - will generate conditional_null rule",
                        field,
                    )
                else:
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
                rule_result = generate_robust_rule(
                    agent_rule,
                    text_data,
                    field,
                    value,
                    field_description,
                    all_fields,
                    max_attempts=args.max_attempts,
                )

                # Unpack the result (rule, prompt_tokens, completion_tokens)
                rule_object, rule_prompt_tokens, rule_completion_tokens = rule_result

                # Track LLM2 usage (rule generator) - accumulate tokens from all attempts
                llm2_calls += 1
                doc_prompt_tokens += rule_prompt_tokens
                doc_completion_tokens += rule_completion_tokens

                logger.debug(
                    "Rule generation token usage - Prompt: %d, Completion: %d",
                    rule_prompt_tokens,
                    rule_completion_tokens,
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

                    if args.cache_filename and args.save_cache_disk:
                        # Save cache to disk immediately after adding rule
                        label_cache.save_to_file_json(
                            filename=args.cache_filename, filepath=args.data_folder
                        )
                        logger.debug(f"✓ Saved cache to: {args.cache_filename}")

                    new_rules_added += 1
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

        # Calculate performance metrics
        accuracy_pct = 0.0
        if "expected_answer" in data:
            accuracy_pct = evaluate_performance(ans, data["expected_answer"])
            logger.info("Document %d accuracy: %.2f%%", doc_idx, accuracy_pct)

        # Store answer for final save
        all_answers.append(
            {
                "idx": doc_idx,
                "expected": data.get("expected_answer", {}),
                "extracted": ans,
            }
        )

        # Determine if we achieved fast path success (100% cache hit)
        fast_path_success = len(failed_fields) == 0

        # Calculate total rules in cache
        total_rules_in_global_cache = sum(
            len(rules)
            for cache in dict_caches.values()
            for rules in cache.fields.values()
        )

        if args.use_wandb and wandb_logger:
            logger.info("Logging wandb metrics for document %d", doc_idx)

            # Update wandb logger with all metrics
            wandb_logger.update_per_doc(
                doc_index=doc_idx,
                doc_id=data.get("filename", f"doc_{doc_idx}"),
                label=data["label"],
                prompt_toks=doc_prompt_tokens,
                completion_toks=doc_completion_tokens,
                price_in=GPT_5_MINI_INPUT_TOKEN_COST,
                price_out=GPT_5_MINI_OUTPUT_TOKEN_COST,
                processing_time=elapsed,
                accuracy_pct=accuracy_pct,
                fields_correct=len(success_fields),
                fields_failed=len(failed_fields),
                total_fields=len(all_fields),
                failed_field_names=failed_fields,
                fast_path_success=fast_path_success,
                new_rules_added=new_rules_added,
                total_rules_in_local_cache=total_rules_in_local_cache,
                total_rules_in_global_cache=total_rules_in_global_cache,
                llm1_calls=llm1_calls,
                llm2_calls=llm2_calls,
            )

            # Log to wandb
            wandb.log(wandb_logger.to_dict())

            # Reset per-doc metrics for next iteration
            wandb_logger.reset_per_doc()

    # Final summary
    logger.info("=" * 80)
    logger.info("Pipeline complete - Processed %d documents", len(processed_dataset))

    # Save answers to disk if requested
    if args.save_ans_disk:
        ans_filepath = os.path.join(args.data_folder, "final_answers.json")
        with open(ans_filepath, "w", encoding="utf-8") as f:
            json.dump(all_answers, f, ensure_ascii=False, indent=4)
        logger.info("Final answers saved to disk: %s", ans_filepath)

    # Upload answers to wandb if requested
    if args.save_ans_wandb and args.use_wandb:
        ans_filepath = os.path.join(args.data_folder, "final_answers.json")
        # Save temporarily to disk first
        with open(ans_filepath, "w", encoding="utf-8") as f:
            json.dump(all_answers, f, ensure_ascii=False, indent=4)
        # Upload to wandb
        wandb.save(ans_filepath)
        logger.info("Final answers uploaded to wandb: %s", ans_filepath)

    # Save cache to disk if requested
    if args.save_cache_disk and args.cache_filename:
        logger.info("Saving all caches to disk")
        for label, cache in dict_caches.items():
            cache.save_to_file_json(
                filename=args.cache_filename, filepath=args.data_folder
            )
            logger.info(
                "Final save - Cache for label '%s' saved to: %s",
                label,
                args.cache_filename,
            )
        logger.info("All caches saved successfully")

    # Upload cache file to wandb if requested
    if args.save_cache_wandb and args.use_wandb and args.cache_filename:
        cache_path = os.path.join(args.data_folder, args.cache_filename)
        wandb.save(cache_path)
        logger.info("Cache file uploaded to wandb: %s", cache_path)

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
