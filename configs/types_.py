from typing import Literal


GROUNDS = Literal[
    "DMX1",
    "HP1",
    "HP2",
    "HP3",
    "LD10",  # Lufeng North - Dushupu. design speed of 160 km/h. Elevation difference 277 m.
    "LD25",  # Lufeng South - Dushupu. design speed of 160 km/h. Elevation difference 277 m.
    "LD50",
]

TRAIN_TYPES = Literal[
    "NL_intercity_VIRM6_",
    "CRH380AL_",
    "HXD1D_",
    # "CRH380AL", "HXD1D", "HXD2",
]

TRAVEL_TIME_SETUP_METHOD = Literal[
    "simple",
    "detailed",
    "conservative",
]