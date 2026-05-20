"""Logging utilities."""

import logging
import sys
from typing import Optional


class RankedLogger(logging.LoggerAdapter):
    """
    Logger that only logs from rank 0 in distributed setting.
    Adapted from the provided code.
    """
    
    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = True,
        extra: Optional[dict] = None
    ):
        logger = logging.getLogger(name)
        super().__init__(logger, extra or {})
        self.rank_zero_only = rank_zero_only
        self._rank = self._get_rank()
    
    def _get_rank(self) -> int:
        """Get current process rank."""
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank()
        except ImportError:
            pass
        return 0
    
    def log(self, level, msg, *args, **kwargs):
        """Log only from rank 0 if rank_zero_only=True."""
        if not self.rank_zero_only or self._rank == 0:
            super().log(level, msg, *args, **kwargs)


def setup_logger(name: str = "trm_reranker", level: int = logging.INFO) -> RankedLogger:
    """Setup logger with formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return RankedLogger(name, rank_zero_only=True)


log = setup_logger()
