"""Compact, deterministic provenance hashes for derived training pools.

All hashes use SHA-256 over UTF-8 bytes of canonical JSON:

* keys are sorted;
* separators are ``,`` and ``:`` with no incidental whitespace;
* non-ASCII characters are preserved;
* selected row ids are sorted ascending before hashing.

The selected ids themselves are too large to embed in every experiment row.
They are exported once per derived variant by the batch runner. Experiment
configuration stores only the count and hash, which are sufficient to verify
the exported id list.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

from app.services.preprocessing import PreprocessState


def canonical_json(value: Any) -> str:
    """Serialise ``value`` deterministically for hashing."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 digest of UTF-8 encoded ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def selected_ids_provenance(ids: Sequence[int]) -> dict[str, Any]:
    """Return the count and canonical hash of a selected-id set."""
    sorted_ids = sorted(int(value) for value in ids)
    return {
        "count": len(sorted_ids),
        "sha256": sha256_text(canonical_json(sorted_ids)),
    }


def preprocessing_state_hash(state: PreprocessState) -> str:
    """Hash the complete fitted preprocessing state."""
    return sha256_text(canonical_json(state.to_dict()))


def variant_identity_hash(provenance: dict[str, Any]) -> str:
    """Hash the complete identity of a derived variant.

    The identity deliberately includes the parent fingerprint, selected-id
    hash and fitted-state hash. Target size or seed alone is not sufficient.
    """
    identity = {
        "parent_fingerprint": provenance["parent_fingerprint"],
        "target_size": provenance["target_size"],
        "training_pool_seed": provenance["training_pool_seed"],
        "selected_id_count": provenance["selected_id_count"],
        "selected_id_sha256": provenance["selected_id_sha256"],
        "preprocessing_mode": provenance["preprocessing_mode"],
        "preprocessing_state_sha256": provenance["preprocessing_state_sha256"],
    }
    return sha256_text(canonical_json(identity))
