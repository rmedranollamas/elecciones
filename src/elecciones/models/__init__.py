"""Statistical electoral models and LOREG Art. 163 D'Hondt allocator."""

from __future__ import annotations

from elecciones.models.dhondt import (
    allocate_batch_vectorized,
    allocate_dhondt_loreg,
    allocate_packed_chunk_inplace,
)
from elecciones.models.ensemble import combine_stage1_results, run_stage1
from elecciones.models.kernel_aggregator import Stage1Result, aggregate_kernel
from elecciones.models.pollster_ratings import (
    compute_pollster_ratings,
    load_pollster_ratings,
    save_pollster_ratings,
)
from elecciones.models.provincial import (
    Stage2DistrictResult,
    Stage2Projection,
    project_2026_provinces,
    project_historical_provinces,
    split_sumar_2023_share,
)
from elecciones.models.state_space import smooth_state_space

__all__ = [
    "Stage1Result",
    "Stage2DistrictResult",
    "Stage2Projection",
    "aggregate_kernel",
    "allocate_batch_vectorized",
    "allocate_dhondt_loreg",
    "allocate_packed_chunk_inplace",
    "combine_stage1_results",
    "compute_pollster_ratings",
    "load_pollster_ratings",
    "project_2026_provinces",
    "project_historical_provinces",
    "run_stage1",
    "save_pollster_ratings",
    "smooth_state_space",
    "split_sumar_2023_share",
]
