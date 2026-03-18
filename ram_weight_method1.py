"""
Method 1: Adaptive partial RAM weight staging.

This module is intentionally standalone and not wired into runtime by default.
It models the policy where only a RAM-budgeted portion of model weights is
kept in RAM, with the remainder fetched from disk/device cache when needed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RamBudgetPlan:
    total_weights_mb: int
    ram_budget_mb: int
    staged_in_ram_mb: int
    streamed_from_storage_mb: int
    strategy: str


class AdaptiveRamBudgetStrategy:
    """Computes a partial staging plan based on available RAM.

    Note: This does not move real model tensors. It defines a policy that can
    be consumed by a lower-level runtime that supports explicit paging.
    """

    MIN_FREE_RAM_MB = 2048

    def build_plan(self, total_weights_mb: int, free_ram_mb: int) -> RamBudgetPlan:
        if total_weights_mb <= 0:
            return RamBudgetPlan(0, 0, 0, 0, "no-model")

        # Keep a floor of free memory for OS + runtime buffers.
        ram_budget_mb = max(0, free_ram_mb - self.MIN_FREE_RAM_MB)
        staged = min(total_weights_mb, ram_budget_mb)
        streamed = max(0, total_weights_mb - staged)

        strategy = "full-ram" if streamed == 0 else "partial-ram-with-streaming"

        return RamBudgetPlan(
            total_weights_mb=total_weights_mb,
            ram_budget_mb=ram_budget_mb,
            staged_in_ram_mb=staged,
            streamed_from_storage_mb=streamed,
            strategy=strategy,
        )
