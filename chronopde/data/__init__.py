"""Data generation and reference simulation utilities."""

from chronopde.data.datasets import (
    HDF5AutoregressiveDataset,
    HDF5RolloutDataset,
    HDF5VelocityDataset,
    NormalizationStats,
    collate_velocity_samples,
    load_normalization,
)
from chronopde.data.initial_conditions import generate_initial_condition
from chronopde.data.manifest import build_dataset_manifest, manifest_hash
from chronopde.data.masks import observation_mask
from chronopde.data.simulator import reaction_diffusion_rhs, simulate_trajectory

__all__ = [
    "HDF5AutoregressiveDataset",
    "HDF5RolloutDataset",
    "HDF5VelocityDataset",
    "NormalizationStats",
    "build_dataset_manifest",
    "collate_velocity_samples",
    "generate_initial_condition",
    "load_normalization",
    "manifest_hash",
    "observation_mask",
    "reaction_diffusion_rhs",
    "simulate_trajectory",
]
