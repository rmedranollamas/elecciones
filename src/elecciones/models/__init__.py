"""Statistical electoral models and LOREG Art. 163 D'Hondt allocator."""

from __future__ import annotations

from elecciones.models.dhondt import (
    allocate_batch_vectorized,
    allocate_dhondt_loreg,
    allocate_packed_chunk_inplace,
)

__all__ = [
    "allocate_batch_vectorized",
    "allocate_dhondt_loreg",
    "allocate_packed_chunk_inplace",
]
