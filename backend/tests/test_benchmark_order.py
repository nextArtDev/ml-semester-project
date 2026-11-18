"""Unit tests for the balanced benchmark-order design.

These tests are pure: no database is required, so the ordering contract can be
checked independently of the experiment runner.
"""

from __future__ import annotations

import pytest

from app.services.benchmark_order import balanced_rounds, positions

KEYS = [f"config-{i}" for i in range(7)]


class TestBalancedRounds:
    def test_every_configuration_runs_once_per_round(self) -> None:
        rounds = balanced_rounds(KEYS, rounds=3, seed=17)
        assert len(rounds) == 3
        for order in rounds:
            assert sorted(order) == sorted(KEYS)

    def test_same_seed_is_reproducible(self) -> None:
        assert balanced_rounds(KEYS, rounds=2, seed=42) == balanced_rounds(
            KEYS, rounds=2, seed=42
        )

    def test_different_seeds_give_different_valid_orders(self) -> None:
        first = balanced_rounds(KEYS, rounds=2, seed=1)
        second = balanced_rounds(KEYS, rounds=2, seed=999)
        assert first != second
        for order in (first[0], second[0]):
            assert sorted(order) == sorted(KEYS)

    def test_every_configuration_is_both_early_and_late(self) -> None:
        rounds = balanced_rounds(KEYS, rounds=3, seed=7)
        half = len(KEYS) // 2
        for key in KEYS:
            occurrences = [order.index(key) + 1 for order in rounds]
            assert min(occurrences) <= half
            assert max(occurrences) > half

    def test_two_rounds_give_distinct_positions(self) -> None:
        rounds = balanced_rounds(KEYS, rounds=2, seed=7)
        for key in KEYS:
            assert rounds[0].index(key) != rounds[1].index(key)

    def test_three_rounds_spread_each_configuration(self) -> None:
        rounds = balanced_rounds(KEYS, rounds=3, seed=7)
        for key in KEYS:
            occurrences = [order.index(key) + 1 for order in rounds]
            assert len(set(occurrences)) == 3
            assert min(occurrences) <= len(KEYS) // 2
            assert max(occurrences) > len(KEYS) // 2

    def test_exact_and_hnsw_are_paired_in_every_round(self) -> None:
        keys = ["exact:-"] + [f"hnsw:{ef}" for ef in (10, 25, 50, 100, 200, 400)]
        for order in balanced_rounds(keys, rounds=2, seed=0):
            assert sorted(order) == sorted(keys)
            assert "exact:-" in order

    def test_positions_are_one_based_and_unique(self) -> None:
        order = balanced_rounds(KEYS, rounds=1, seed=3)[0]
        mapping = positions(order)
        assert len(mapping) == len(KEYS)
        assert sorted(mapping.values()) == list(range(1, len(KEYS) + 1))

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            balanced_rounds([], rounds=1, seed=0)
        with pytest.raises(ValueError):
            balanced_rounds(["a", "a"], rounds=1, seed=0)
        with pytest.raises(ValueError):
            balanced_rounds(KEYS, rounds=0, seed=0)
