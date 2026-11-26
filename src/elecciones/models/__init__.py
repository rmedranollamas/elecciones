"""Statistical electoral models and LOREG Art. 163 D'Hondt allocator."""

from __future__ import annotations

from elecciones.models.covariance import (
    HierarchicalCovarianceConfig,
    SimulationArrays,
    estimate_empirical_error_hierarchy,
    run_hierarchical_monte_carlo,
)
from elecciones.models.dhondt import (
    allocate_batch_vectorized,
    allocate_dhondt_loreg,
    allocate_packed_chunk_inplace,
)
from elecciones.models.ensemble import combine_stage1_results, run_stage1
from elecciones.models.investiture import (
    compute_investiture_summary,
    compute_majority_histogram,
    evaluate_first_ballot,
    evaluate_second_ballot,
)
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
from elecciones.models.simulator import run_forecast_simulation
from elecciones.models.state_space import smooth_state_space

__all__ = [
    "HierarchicalCovarianceConfig",
    "SimulationArrays",
    "Stage1Result",
    "Stage2DistrictResult",
    "Stage2Projection",
    "aggregate_kernel",
    "allocate_batch_vectorized",
    "allocate_dhondt_loreg",
    "allocate_packed_chunk_inplace",
    "combine_stage1_results",
    "compute_investiture_summary",
    "compute_majority_histogram",
    "compute_pollster_ratings",
    "estimate_empirical_error_hierarchy",
    "evaluate_first_ballot",
    "evaluate_second_ballot",
    "load_pollster_ratings",
    "project_2026_provinces",
    "project_historical_provinces",
    "run_forecast_simulation",
    "run_hierarchical_monte_carlo",
    "run_stage1",
    "save_pollster_ratings",
    "smooth_state_space",
    "split_sumar_2023_share",
]
