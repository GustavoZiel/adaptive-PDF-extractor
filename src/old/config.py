"""Configuration and constants for the extraction pipeline."""

import os
from dataclasses import dataclass
from typing import Optional

# ============================================================================
# SECTION 1: Model Pricing Constants
# ============================================================================
# Source: https://platform.openai.com/docs/models/gpt-4-mini
GPT_4_MINI_INPUT_COST = 0.15 / 1_000_000  # $0.15 per 1M input tokens
GPT_4_MINI_OUTPUT_COST = 0.60 / 1_000_000  # $0.60 per 1M output tokens

# Note: Update these if using GPT-5-mini or other models
GPT_5_MINI_INPUT_COST = 0.25 / 1_000_000  # $0.25 per 1M input tokens
GPT_5_MINI_OUTPUT_COST = 2.00 / 1_000_000  # $2.00 per 1M output tokens


# ============================================================================
# SECTION 2: Configuration Dataclass
# ============================================================================


@dataclass
class Config:
    """Runtime configuration for the extraction pipeline.

    This class encapsulates all CLI arguments and runtime settings
    used throughout the extraction pipeline.
    """

    # Input/Output
    data_folder: str
    dataset_filename: str

    # Processing
    max_attempts: int
    cache_filename: Optional[str]

    # Logging
    log_level: str

    # WandB Integration
    use_wandb: bool
    wandb_project: str
    wandb_run_name: Optional[str]

    # Save Options
    save_ans_disk: bool
    save_ans_wandb: bool
    save_cache_disk: bool
    save_cache_wandb: bool

    @classmethod
    def from_args(cls, args):
        """Create Config instance from CLI arguments.

        Args:
            args: Arguments object from tyro.cli(Args)

        Returns:
            Config instance with all settings
        """
        return cls(
            data_folder=str(args.data_folder),
            dataset_filename=args.dataset_filename,
            max_attempts=args.max_attempts,
            cache_filename=args.cache_filename,
            log_level=args.log_level,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            save_ans_disk=args.save_ans_disk,
            save_ans_wandb=args.save_ans_wandb,
            save_cache_disk=args.save_cache_disk,
            save_cache_wandb=args.save_cache_wandb,
        )

    def get_cache_path(self) -> Optional[str]:
        """Get full path to cache file if configured.

        Returns:
            Full path to cache file, or None if not configured
        """
        if self.cache_filename:
            return os.path.join(self.data_folder, self.cache_filename)
        return None

    def get_dataset_path(self) -> str:
        """Get full path to dataset file.

        Returns:
            Full path to dataset file
        """
        return os.path.join(self.data_folder, self.dataset_filename)
