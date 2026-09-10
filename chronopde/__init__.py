"""ChronoPDE scientific machine-learning package."""

from chronopde.config import ProjectConfig, load_config
from chronopde.contracts import (
    GridSpec,
    PhysicalParameters,
    SimulationDiagnostics,
    SimulationResult,
)
from chronopde.data import generate_initial_condition, reaction_diffusion_rhs, simulate_trajectory
from chronopde.numerics import build_grid, build_neumann_laplacian

__all__ = [
    "GridSpec",
    "PhysicalParameters",
    "ProjectConfig",
    "SimulationDiagnostics",
    "SimulationResult",
    "build_grid",
    "build_neumann_laplacian",
    "generate_initial_condition",
    "load_config",
    "reaction_diffusion_rhs",
    "simulate_trajectory",
]
__version__ = "0.1.0"
