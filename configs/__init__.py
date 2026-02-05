# configs/__init__.py
"""Configuration module for VAO-TC project."""

from configs.types_ import GROUNDS, TRAIN_TYPES, TRAVEL_TIME_SETUP_METHOD
from configs.ground import GroundConfigBase, GroundConfigM2
from configs.train import TrainConfig

__all__ = [
    # Types
    "GROUNDS",
    "TRAIN_TYPES", 
    "TRAVEL_TIME_SETUP_METHOD",
    # Ground configs
    "GroundConfigBase",
    "GroundConfigM2",
    # Train config
    "TrainConfig",
]