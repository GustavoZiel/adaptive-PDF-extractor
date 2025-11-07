"""Main extraction pipeline CLI.

This is the entry point for the adaptive PDF extraction system.
It orchestrates the entire extraction workflow using modular components.
"""

import logging
import time
from collections import defaultdict

import tyro
from dotenv import load_dotenv
from pydantic import BaseModel, DirectoryPath, Field

import wandb
from cache import Cache, load_dict_cache_json, save_dict_cache
from data import create_pydantic_model, format_dict, process_dataset, read_dataset
from llm import (
    EXTRACTION_PROMPT,
    create_extraction_agent,
    create_rule_agent,
    init_model,
)
from logger import get_logger, set_global_log_level
from metrics import (
    MetricsTracker,
    evaluate_performance,
    save_cache,
    save_results,
    setup_wandb,
)
from pipeline import extract_with_cache, extract_with_llm, generate_rules_for_fields
from rule import Rule

load_dotenv()

logger = None

# ============================================================================
# Model Pricing Constants
# ============================================================================

# Source: https://platform.openai.com/docs/models/gpt-5-mini
GPT_5_MINI_INPUT_COST = 0.25 / 1_000_000  # $0.25 per 1M input tokens
GPT_5_MINI_OUTPUT_COST = 2.00 / 1_000_000  # $2.00 per 1M output tokens


# ============================================================================
# CLI Arguments
# ============================================================================


class Args(BaseModel):
    """CLI arguments for the extraction pipeline."""

    data_folder: DirectoryPath = Field(
        description="Path to the data folder containing PDFs and dataset."
    )
    dataset_filename: str = Field(
        description="Name of the dataset JSON file within the data folder."
    )
    max_attempts: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of rule generation attempts per field.",
    )
    cache_filename: str | None = Field(
        None,
        description="Name of the cache file within the data folder (optional).",
    )
    use_wandb: bool = Field(
        default=False,
        description="Enable Weights & Biases (wandb) logging for experiment tracking.",
    )
    wandb_project: str = Field(
        default="adaptive-pdf-extractor",
        description="Wandb project name for logging.",
    )
    wandb_run_name: str | None = Field(
        default=None,
        description="Optional custom name for the wandb run.",
    )
    save_ans_disk: bool = Field(
        default=True,
        description="Save final answers to disk as JSON file.",
    )
    save_ans_wandb: bool = Field(
        default=True,
        description="Upload final answers to wandb as run file.",
    )
    save_cache_disk: bool = Field(
        default=True,
        description="Save cache to disk as JSON file after each rule generation.",
    )
    save_cache_wandb: bool = Field(
        default=True,
        description="Upload cache to wandb as run file at the end.",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    max_retries: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Maximum number of retries for LLM API calls.",
    )
    timeout: int = Field(
        default=90,
        ge=5,
        le=120,
        description="Timeout for each LLM API calls in seconds.",
    )
    use_cache: bool = Field(
        default=True,
        description="Enable cache-based extraction. Disable for LLM-only mode.",
    )


# ============================================================================
# Main Pipeline
# ============================================================================


def main(args: Args):
    """Main extraction pipeline orchestration.

    This function coordinates the entire workflow:
    1. Initialize LLM model and load dataset
    2. For each document:
       a. Try cache-based extraction (fast path)
       b. Use LLM extraction for failed fields (slow path)
       c. Generate and cache new rules for future use
    3. Track metrics and save results
    """
    # Convert Args to Config
    # config = Config.from_args(args)
    config = args
    logger.info("Starting extraction pipeline")
    logger.debug("Configuration: %s", config)

    # Initialize LLM model
    model = init_model(max_retries=config.max_retries, timeout=config.timeout)
    logger.info("LLM model initialized successfully")

    # Load and process dataset
    dataset = read_dataset(config.dataset_filename, config.data_folder)
    processed_dataset = process_dataset(dataset, config.data_folder)
    logger.info("Dataset loaded and processed: %d documents", len(processed_dataset))

    # Initialize cache system (one cache per document label/type)
    dict_caches = defaultdict(Cache)

    # Load global cache if provided and cache is enabled
    if config.use_cache and config.cache_filename:
        global_loaded_cache = load_dict_cache_json(
            config.cache_filename, config.data_folder
        )
    else:
        global_loaded_cache = None
        if not config.use_cache:
            logger.info("Cache disabled - operating in LLM-only mode")

    # Initialize metrics tracking
    metrics_tracker = None
    if config.use_wandb:
        metrics_tracker = MetricsTracker()
        setup_wandb(config)
        logger.info("WandB tracking initialized")

    # Collect all answers for final save
    all_answers = []

    # Process each document in the dataset
    for doc_idx, data in enumerate(processed_dataset, 1):
        start_time = time.time()

        text_data = data.get("pdf_text", "")
        all_fields = list(data["extraction_schema"].keys())

        logger.info("=" * 80)
        logger.info(
            "Processing document %d/%d - Label: '%s'",
            doc_idx,
            len(processed_dataset),
            data["label"],
        )

        # Use loaded cache for this label (only on first encounter)
        if global_loaded_cache and data["label"] not in dict_caches:
            dict_caches[data["label"]] = global_loaded_cache[data["label"]]
            logger.debug("Using loaded cache for label '%s'", data["label"])

        label_cache = dict_caches[data["label"]]

        # ====================================================================
        # STEP 1: Cache-Based Extraction (Fast Path)
        # ====================================================================

        ans = {}
        success_fields = []
        failed_fields = []

        if config.use_cache:
            # Try cache-based extraction first
            ans, success_fields, failed_fields = extract_with_cache(
                label_cache, data["label"], text_data, all_fields
            )

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
        else:
            # Cache disabled - all fields need LLM extraction
            logger.info("Cache disabled - using LLM-only extraction mode")
            failed_fields = all_fields.copy()

        # ====================================================================
        # STEP 2: LLM Extraction for Failed Fields (Slow Path)
        # ====================================================================

        doc_prompt_tokens = 0
        doc_completion_tokens = 0
        llm1_calls = 0
        llm2_calls = 0
        new_rules_added = 0

        if failed_fields:
            logger.info("Extracting %d failed fields using LLM", len(failed_fields))

            # Create dynamic Pydantic model for failed fields only
            failed_schema = {f: data["extraction_schema"][f] for f in failed_fields}
            pydantic_model = create_pydantic_model(failed_schema)

            # Create extraction agent
            agent = create_extraction_agent(model, pydantic_model)

            # Extract with LLM
            (
                llm_extracted,
                extraction_success,
                extractor_prompt_toks,
                extractor_completion_toks,
            ) = extract_with_llm(
                agent,
                text_data,
                data["extraction_schema"],
                failed_fields,
                EXTRACTION_PROMPT,
            )
            ans.update(llm_extracted)
            doc_prompt_tokens += extractor_prompt_toks
            doc_completion_tokens += extractor_completion_toks
            llm1_calls = 1

            if extraction_success:
                logger.info("LLM extraction complete:\n%s", format_dict(llm_extracted))

                # ================================================================
                # STEP 3: Generate and Cache Rules (only if cache is enabled)
                # ================================================================

                if config.use_cache:
                    logger.info(
                        "Generating rules for %d extracted fields", len(llm_extracted)
                    )

                    # Create rule generation agent
                    agent_rule = create_rule_agent(model, Rule)

                    # Define save callback
                    # (saves cache immediately to disk after each rule)
                    def save_cache_callback():
                        if config.save_cache_disk:
                            save_dict_cache(dict_caches, config)

                    # Generate rules with validation loop
                    (
                        new_rules_added,
                        rule_prompt_toks,
                        rule_completion_toks,
                        llm_calls,
                    ) = generate_rules_for_fields(
                        agent_rule,
                        llm_extracted,
                        text_data,
                        data["extraction_schema"],
                        all_fields,
                        label_cache,
                        data["label"],
                        config.max_attempts,
                        save_cache_callback,
                    )

                    doc_prompt_tokens += rule_prompt_toks
                    doc_completion_tokens += rule_completion_toks
                    llm2_calls = llm_calls

                    logger.info("Generated %d new rules", new_rules_added)
                else:
                    logger.debug("Cache disabled - skipping rule generation")
            else:
                logger.warning(
                    "LLM extraction failed (timeout or error). "
                    "Skipping rule generation for %d fields: %s",
                    len(failed_fields),
                    failed_fields,
                )
        else:
            logger.info("All fields extracted from cache (100%% hit rate)")
            if config.cache_filename and config.save_cache_disk:
                save_dict_cache(dict_caches, config)
                logger.debug("Cache saved after successful full extraction")

        # ====================================================================
        # STEP 4: Evaluate Performance and Track Metrics
        # ====================================================================

        elapsed_time = time.time() - start_time

        # Evaluate accuracy if ground truth available
        accuracy_pct = 0.0
        if "expected_answer" in data:
            accuracy_pct = evaluate_performance(ans, data["expected_answer"])

        # Collect answer
        all_answers.append(
            {
                "idx": doc_idx,
                "label": data["label"],
                "expected": data.get("expected_answer", {}),
                "extracted": ans,
            }
        )

        # Update metrics if tracking enabled
        if metrics_tracker:
            total_rules_in_global_cache = sum(
                len(rules_list)
                for cache in dict_caches.values()
                for rules_list in cache.fields.values()
            )

            metrics_tracker.update_per_doc(
                doc_index=doc_idx,
                doc_id=data.get("filename", f"doc_{doc_idx}"),
                label=data["label"],
                prompt_toks=doc_prompt_tokens,
                completion_toks=doc_completion_tokens,
                price_in=GPT_5_MINI_INPUT_COST,
                price_out=GPT_5_MINI_OUTPUT_COST,
                processing_time=elapsed_time,
                accuracy_pct=accuracy_pct,
                fields_correct=len(success_fields),
                fields_failed=len(failed_fields),
                total_fields=len(all_fields),
                failed_field_names=failed_fields,
                fast_path_success=(len(failed_fields) == 0),
                new_rules_added=new_rules_added,
                total_rules_in_local_cache=sum(
                    len(r) for r in label_cache.fields.values()
                ),
                total_rules_in_global_cache=total_rules_in_global_cache,
                llm1_calls=llm1_calls,
                llm2_calls=llm2_calls,
            )

            # Log to WandB
            wandb.log(metrics_tracker.to_dict())
            metrics_tracker.reset_per_doc()

        logger.info(
            "Document %d complete - Time: %.2fs, Accuracy: %.1f%%",
            doc_idx,
            elapsed_time,
            accuracy_pct,
        )

    # ========================================================================
    # Final Save and Cleanup
    # ========================================================================

    logger.info("=" * 80)
    logger.info("Pipeline complete - Processed %d documents", len(processed_dataset))

    # Save final results
    save_results(all_answers, config)

    # Save final cache (only if cache is enabled)
    if config.use_cache:
        save_cache(dict_caches, config)

    # Finish WandB run
    if config.use_wandb:
        wandb.finish()
        logger.info("WandB run finished")

    logger.info("Extraction pipeline completed successfully!")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    # Parse CLI arguments
    args = tyro.cli(Args)

    # Setup logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    set_global_log_level(log_level)
    logger = get_logger(__name__, level=log_level)
    globals()["logger"] = logger

    # Run pipeline
    main(args)
