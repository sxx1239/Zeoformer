"""Pydantic model for default configuration and validation."""
"""Implementation based on the template of ALIGNN,Matformer."""

import subprocess
from typing import Optional, Union
import os
from pydantic import root_validator

# vfrom pydantic import Field, root_validator, validator
from pydantic.typing import Literal
from pddformer.models.pddformer import PDDFormerConfig, BaseSettings

# from typing import List

try:
    VERSION = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    )
except Exception as exp:
    VERSION = "NA"
    pass


FEATURESET_SIZE = {"basic": 11, "atomic_number": 1, "cfid": 438, "cgcnn": 92}


TARGET_ENUM = Literal[
    "formation_energy_peratom",
    "optb88vdw_bandgap",
    "bulk_modulus_kv",
    "shear_modulus_gv",
    "mbj_bandgap",
    "slme",
    "spillage",
    "optb88vdw_total_energy",
    "ehull",
    "gap pbe",
    "e_form",
    "e_hull",
    "band_gap",
    "bulk modulus",
    "shear modulus",
    "bandgap",
    "energy_total",
]


class TrainingConfig(BaseSettings):
    """Training config defaults and validation."""

    version: str = VERSION

    # dataset configuration
    dataset: Literal[
        "dft_3d",
        "megnet",
    ] = "dft_3d"
    target: TARGET_ENUM = "formation_energy_peratom"
    atom_features: Literal["basic", "atomic_number", "cfid", "cgcnn"] = "cgcnn"
    neighbor_strategy: Literal["k-nearest", "voronoi", "pairwise-k-nearest"] = "k-nearest"
    id_tag: Literal["jid", "id", "_oqmd_entry_id"] = "jid"

    # logging configuration

    # training configuration
    random_seed: Optional[int] = 123
    classification_threshold: Optional[float] = None
    n_val: Optional[int] = None
    n_test: Optional[int] = None
    n_train: Optional[int] = None
    train_ratio: Optional[float] = 0.8
    val_ratio: Optional[float] = 0.1
    test_ratio: Optional[float] = 0.1
    target_multiplication_factor: Optional[float] = None
    epochs: int = 300
    batch_size: int = 64
    weight_decay: float = 0
    learning_rate: float = 1e-2
    filename: str = "sample"
    warmup_steps: int = 2000
    criterion: Literal["mse", "l1", "poisson", "zig"] = "l1"
    optimizer: Literal["adamw", "sgd"] = "adamw"
    scheduler: Literal["onecycle", "none", "step", "polynomial"] = "onecycle"
    pin_memory: bool = False
    save_dataloader: bool = False
    write_checkpoint: bool = True
    write_predictions: bool = True
    store_outputs: bool = True
    progress: bool = True
    log_tensorboard: bool = False
    standard_scalar_and_pca: bool = False
    use_canonize: bool = True
    num_workers: int = 2
    cutoff: float = 8.0
    max_neighbors: int = 16
    keep_data_order: bool = False
    distributed: bool = False
    n_early_stopping: Optional[int] = None  # typically 50
    output_dir: str = os.path.abspath(".")  # typically 50
    matrix_input: bool = False
    pyg_input: bool = False

    # model configuration
    model = PDDFormerConfig(name="pddformer")
    print(model)
    @root_validator()
    def set_input_size(cls, values):
        """Automatically configure node feature dimensionality."""
        values["model"].cgcnn_features = FEATURESET_SIZE[
            values["atom_features"]
        ]

        return values
