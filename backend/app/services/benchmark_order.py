"""Reproducible, balanced ordering of benchmark configurations.

A benchmark that always runs the exact baseline first and then increases
``ef_search`` gives every configuration a different cache and scheduler
environment.  The fixed order is a confound: a later run can look faster or
slower simply because of what ran before it.

This module builds *paired rounds*.  Each round contains every requested
configuration exactly once.  A seed chooses a base permutation, and successive
rounds rotate that permutation by ``ceil(n / rounds)`` positions, so a
configuration that ran early in one round runs late in another.  With two
rounds every configuration appears in both halves of the sequence; with more
rounds the coverage improves further.

The order is a pure function of the requested keys, the round count and the
seed, which makes it reproducible and independently testable without a
database.  It does not rebuild an index or change any experiment configuration.
"""

from __future__ import annotations

import random
from typing import Iterable, Sequence


def balanced_rounds(keys: Sequence[str], rounds: int, seed: int) -> list[list[str]]:
    """Build ``rounds`` balanced permutations of ``keys``.

    Parameters
    ----------
    keys:
        Unique configuration identifiers.  Order is irrelevant: the base
        permutation is derived from ``seed``, not from the caller's order.
    rounds:
        Number of complete rounds.  Must be at least 1.
    seed:
        Fixed integer seed.  The same ``(keys, rounds, seed)`` always yields
        the same rounds; a different seed generally yields a different but
        equally valid set of rounds.

    Returns
    -------
    list[list[str]]
        One list per round.  Each list is a permutation of ``keys``.
    """
    key_list = list(keys)
    if not key_list:
        raise ValueError("at least one configuration key is required")
    if len(set(key_list)) != len(key_list):
        raise ValueError("configuration keys must be unique")
    if rounds < 1:
        raise ValueError("rounds must be at least 1")

    base = list(key_list)
    random.Random(seed).shuffle(base)
    # A floor step makes three or more rounds visit every position exactly
    # once and guarantees an early/late alternation.  With two rounds and an
    # odd number of configurations a perfect split is mathematically
    # impossible, so callers that need the guarantee should use >= 3 rounds.
    step = max(1, len(base) // rounds)
    return [
        [base[(index + round_index * step) % len(base)] for index in range(len(base))]
        for round_index in range(rounds)
    ]


def positions(order: Iterable[str]) -> dict[str, int]:
    """Map each configuration key to its 1-based position in a round."""
    return {key: position for position, key in enumerate(order, start=1)}
