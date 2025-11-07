"""Metrics tracking, WandB integration, and performance evaluation.

This module handles all metrics collection, experiment tracking via WandB/Weave,
and performance evaluation for the extraction pipeline.
"""

import json
import os
from dataclasses import dataclass, field
from typing import List

import weave

import wandb
from cache import save_dict_cache
from data import format_dict, get_json_filename
from logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# SECTION 1: Metrics Tracker
# ============================================================================


@dataclass
class MetricsTracker:
    """Track metrics across documents (formerly WandbLogProperties).

    This class maintains both cumulative metrics (totals/averages) and
    per-document metrics (reset after each document).
    """

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
    llm1_extractor_calls_per_doc: int = 0
    llm2_generator_calls_per_doc: int = 0
    total_per_doc: int = 0
    total_llm_calls: int = 0
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
        """Update all per-document stats and roll up to totals.

        Args:
            doc_index: Current document index
            doc_id: Document identifier
            label: Document type/label
            prompt_toks: Number of input tokens used
            completion_toks: Number of output tokens generated
            price_in: Cost per input token
            price_out: Cost per output token
            processing_time: Time taken to process document (seconds)
            accuracy_pct: Extraction accuracy percentage
            fields_correct: Number of correctly extracted fields
            fields_failed: Number of failed field extractions
            total_fields: Total number of fields in document
            failed_field_names: List of failed field names
            fast_path_success: Whether all fields extracted from cache
            new_rules_added: Number of new rules generated
            total_rules_in_local_cache: Rules in this label's cache
            total_rules_in_global_cache: Rules across all labels
            llm1_calls: Number of extraction LLM calls
            llm2_calls: Number of rule generation LLM calls
        """
        self.doc_index = doc_index
        self.doc_id = doc_id
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
        self.llm1_extractor_calls_per_doc = llm1_calls
        self.llm2_generator_calls_per_doc = llm2_calls
        self.total_per_doc = (
            self.llm1_extractor_calls_per_doc + self.llm2_generator_calls_per_doc
        )
        self.llm1_extractor_calls += self.llm1_extractor_calls_per_doc
        self.llm2_generator_calls += self.llm2_generator_calls_per_doc
        self.total_llm_calls += self.total_per_doc

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
        """Return a dict for wandb logging, excluding internal fields.

        Returns:
            Dictionary with organized metrics for WandB logging
        """
        data = {
            # Document identification
            "doc_index": self.doc_index,
            "label": self.label,
            # Performance metrics
            "performance/per_doc": self.performance_per_doc,
            "performance/avg": self.avg_performance,
            "performance/fields_correct": self.fields_correct_count,
            "performance/fields_failed": self.fields_failed_count,
            "performance/failed_field_names": self.failed_field_names,
            # Time metrics
            "time/processing_time_sec": self.processing_time_sec,
            "time/avg_processing_time_sec": self.avg_processing_time_sec,
            "time/total_elapsed": self.total_elapsed,
            # Cost metrics
            "cost/per_doc": self.cost,
            "cost/total": self.total_cost,
            "cost/avg_per_doc": self.avg_cost_per_doc,
            # Token usage
            "tokens/prompt": self.prompt_tokens,
            "tokens/completion": self.completion_tokens,
            "tokens/total_prompt": self.total_prompt_tokens,
            "tokens/total_completion": self.total_completion_tokens,
            # Cache & Healing metrics
            "cache/hit_rate": self.cache_hit_rate,
            "cache/fast_path_success": self.fast_path_success,
            "cache/fast_path_total_success": self._sum_cache_hits,
            "cache/new_rules_added": self.new_rules_added,
            "cache/total_rules_in_local_cache": self.total_rules_in_local_cache,
            "cache/total_rules_in_global_cache": self.total_rules_in_global_cache,
            # LLM calls tracking
            "llm_calls/extractor_total": self.llm1_extractor_calls,
            "llm_calls/generator_total": self.llm2_generator_calls,
            "llm_calls/total": self.total_llm_calls,
            "llm_calls/extractor_per_doc": self.llm1_extractor_calls_per_doc,
            "llm_calls/generator_per_doc": self.llm2_generator_calls_per_doc,
            "llm_calls/total_per_doc": self.total_per_doc,
        }

        return data


# ============================================================================
# SECTION 2: Performance Evaluation
# ============================================================================


def evaluate_performance(extracted: dict, expected: dict) -> float:
    """Evaluate extraction accuracy by comparing extracted vs expected values.

    Args:
        extracted: Dictionary of extracted field values
        expected: Dictionary of expected (ground truth) field values

    Returns:
        Percentage of correctly extracted fields (0-100)
    """
    num_fields_expected = len(expected)
    if num_fields_expected == 0:
        logger.warning("No expected fields to evaluate against")
        return 0.0

    num_correct = sum(
        1 for k, v in expected.items() if k in extracted and extracted[k] == v
    )

    pct_correct = (num_correct / num_fields_expected) * 100.0

    comparison = {
        key: {"expected": ref_value, "extracted": extracted.get(key)}
        for key, ref_value in expected.items()
    }
    logger.debug("Comparison:\n%s", format_dict(comparison))

    logger.info(
        "Performance: %d/%d fields correct (%.2f%%)",
        num_correct,
        num_fields_expected,
        pct_correct,
    )

    return pct_correct


# ============================================================================
# SECTION 3: WandB Setup and Configuration
# ============================================================================


def setup_wandb(config):
    """Initialize Weights & Biases and Weave for experiment tracking.

    Args:
        config: Config instance with wandb settings

    Returns:
        wandb Run object, or None if WandB not configured
    """
    wandb_api_key = os.getenv("WANDB_API_KEY")

    if not wandb_api_key or wandb_api_key == "your_wandb_api_key_here":
        logger.warning("WANDB_API_KEY not set in .env file.")
        logger.warning("Proceeding without wandb logging...")
        return None

    # Initialize WandB
    wandb_run = wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=vars(config) if hasattr(config, "__dict__") else config,
    )

    # Initialize Weave (LangChain tracing)
    weave.init(project_name=config.wandb_project)

    # Define metrics
    _define_wandb_metrics()

    logger.info(
        "Wandb and Weave initialized - Project: %s, Run: %s",
        config.wandb_project,
        wandb_run.name if wandb_run else "N/A",
    )

    return wandb_run


def _define_wandb_metrics():
    """Define all wandb metrics with proper step and summary aggregations.

    This ensures WandB properly tracks and visualizes metrics over time.
    """
    # Define doc_index as the primary x-axis
    wandb.define_metric("doc_index")
    wandb.define_metric("*", step_metric="doc_index")

    # 1. Performance Metrics
    wandb.define_metric("performance/per_doc", summary="mean")
    wandb.define_metric("performance/per_doc", summary="max")
    wandb.define_metric("performance/per_doc", summary="min")
    wandb.define_metric("performance/per_doc", summary="last")
    wandb.define_metric("performance/avg", summary="last")

    # 2. Cost & Efficiency Metrics
    wandb.define_metric("time/processing_time_sec", summary="mean")
    wandb.define_metric("time/processing_time_sec", summary="max")
    wandb.define_metric("time/processing_time_sec", summary="min")
    wandb.define_metric("time/avg_processing_time_sec", summary="last")

    wandb.define_metric("cost/per_doc", summary="mean")
    wandb.define_metric("cost/total", summary="last")
    wandb.define_metric("cost/avg_per_doc", summary="last")

    wandb.define_metric("tokens/total_prompt", summary="last")
    wandb.define_metric("tokens/total_completion", summary="last")

    # 3. Cache & Healing Metrics
    wandb.define_metric("cache/fast_path_success", summary="mean")
    wandb.define_metric("cache/hit_rate", summary="last")

    wandb.define_metric("cache/total_rules_in_local_cache", summary="last")
    wandb.define_metric("cache/total_rules_in_global_cache", summary="last")

    logger.debug("WandB metrics defined successfully")


# ============================================================================
# SECTION 4: Results Saving
# ============================================================================


def save_results(all_answers: list, config):
    """Save extraction results to disk and/or WandB.

    Args:
        all_answers: List of answer dictionaries with idx, expected, extracted
        config: Config instance with save settings
    """
    if config.use_cache:
        filename = config.dataset_filename + "_with_cache" + "_result.json"
    else:
        filename = config.dataset_filename + "_without_cache" + "_result.json"

    filepath = os.path.join(config.data_folder, filename)
    os.makedirs(config.data_folder, exist_ok=True)

    if config.save_ans_disk:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(all_answers, f, ensure_ascii=False, indent=4)

        logger.info("Results saved to disk: %s", filepath)

    if config.save_ans_wandb and config.use_wandb:
        # Save to disk first (WandB uploads from disk)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(all_answers, f, ensure_ascii=False, indent=4)

        try:
            wandb.save(filepath)
            logger.info("Results uploaded to wandb: %s", filepath)
        except Exception as e:
            logger.error("Failed to upload results to wandb: %s", e)


def save_cache(dict_caches: dict, config):
    """Save all caches to a JSON file.

    Args:
        dict_caches: Dictionary mapping labels to Cache instances
        config: Config instance with save settings
    """
    if not dict_caches:
        logger.warning("No caches to save.")
        return

    if config.save_cache_disk:
        save_dict_cache(dict_caches, config)
        # filename = get_json_filename(config.cache_filename)
        # filepath = os.path.join(config.data_folder, filename)
        # os.makedirs(config.data_folder, exist_ok=True)

        # data_to_save = {label: cache.to_dict() for label, cache in dict_caches.items()}

        # try:
        #     with open(filepath, "w", encoding="utf-8") as f:
        #         json.dump(data_to_save, f, ensure_ascii=False, indent=4)

        #     logger.info("Caches saved to disk: %s", filepath)
        # except Exception as e:
        #     logger.error("Failed to save caches to disk: %s", e)

    if config.save_cache_wandb and config.use_wandb:
        if not config.cache_filename:
            cache_filename = config.dataset_filename + "_cache.json"
        else:
            cache_filename = get_json_filename(config.cache_filename)

        filepath = os.path.join(config.data_folder, cache_filename)
        os.makedirs(config.data_folder, exist_ok=True)

        data_to_save = {label: cache.to_dict() for label, cache in dict_caches.items()}

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=4)

            wandb.save(filepath)
            logger.info("Caches uploaded to wandb: %s", filepath)
        except Exception as e:
            logger.error("Failed to upload caches to wandb: %s", e)
