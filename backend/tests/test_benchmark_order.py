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


MATRIX = ["exact:-"] + [f"hnsw:{ef}" for ef in (10, 25, 50, 100, 200, 400)]


class TestActualBenchmarkMatrix:
    """Exhaustive checks for the supported 7-configuration, 3-round matrix."""

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 12345, -17, 10**9])
    def test_positions_are_balanced_for_supported_matrix(self, seed: int) -> None:
        rounds = balanced_rounds(MATRIX, rounds=3, seed=seed)
        for order in rounds:
            assert sorted(order) == sorted(MATRIX)

        # Each configuration occupies three different positions.
        for key in MATRIX:
            positions_used = [order.index(key) + 1 for order in rounds]
            assert len(set(positions_used)) == 3

        # Each position is occupied by three different configurations.
        for position in range(1, len(MATRIX) + 1):
            occupants = [order[position - 1] for order in rounds]
            assert len(set(occupants)) == 3

        # Every configuration has an early and a late slot.
        for key in MATRIX:
            positions_used = [order.index(key) + 1 for order in rounds]
            assert min(positions_used) <= len(MATRIX) // 2
            assert max(positions_used) > len(MATRIX) // 2

    @pytest.mark.parametrize("seed", [0, 42, -7])
    def test_exact_is_not_fixed_in_a_position_relative_to_hnsw(
        self, seed: int
    ) -> None:
        rounds = balanced_rounds(MATRIX, rounds=3, seed=seed)
        exact_positions = [order.index("exact:-") + 1 for order in rounds]
        assert len(set(exact_positions)) > 1
        assert not (max(exact_positions) <= 2 or min(exact_positions) >= 7)
        for key in MATRIX:
            if key == "exact:-":
                continue
            assert [order.index(key) + 1 for order in rounds] != exact_positions

    def test_seed_changes_orders_for_supported_matrix(self) -> None:
        first = balanced_rounds(MATRIX, rounds=3, seed=1)
        second = balanced_rounds(MATRIX, rounds=3, seed=2)
        assert first != second
        for order in (first[0], second[0]):
            assert sorted(order) == sorted(MATRIX)
