"""Logging setup utilities for training.

Provides functions for setting up logging in distributed training scenarios.
"""

import logging
from pathlib import Path


def setup_logging(
    rank: int = 0,
    output_dir: Path | None = None,
    log_level: int | None = None,
) -> None:
    """Setup logging configuration for distributed training.

    Sets up logging with appropriate level based on rank:
    - Rank 0: INFO level with optional file handler
    - Other ranks: WARNING level (reduces noise in multi-GPU)

    Args:
        rank: Process rank in distributed training (0 = main process)
        output_dir: Optional directory for log file (rank 0 only)
        log_level: Optional explicit log level (overrides rank-based default)

    Example:
        >>> from wf_train.utils import setup_logging
        >>> setup_logging(rank=0, output_dir=Path("./outputs"))
    """
    # Determine log level
    if log_level is not None:
        level = log_level
    else:
        level = logging.INFO if rank == 0 else logging.WARNING

    # Configure handlers
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if output_dir is not None and rank == 0:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(output_dir / "train.log"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,  # Override any existing configuration
    )
