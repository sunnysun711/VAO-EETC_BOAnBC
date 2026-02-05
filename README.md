# Code Repository for "Branch Outer Approximate and Benders Cuts Algorithm for Energy-Efficient Railway Vertical Alignment Optimization"

**Author:** Yichen Sun

## Overview

This repository contains the implementation code for the manuscript on energy-efficient railway vertical alignment optimization using the Branch Outer Approximation and Benders Cuts (BOAnBC) algorithm.

## Project Structure

```text
.
├── configs/                      # Configuration modules
│   ├── __init__.py
│   ├── ground.py                 # Ground configuration classes
│   ├── train.py                  # Train configuration classes
│   └── types_.py                 # Type definitions
│
├── data/                         # Data files
│   ├── grounds/                  # Ground profile data
│   └── trains/                   # Train parameter data
│
├── models/                       # Optimization models
│   ├── tc_socp.py                # Train control SOCP model
│   └── vao2.py                   # Vertical alignment optimization model (M2)
│
├── solvers/                      # Solution algorithms
│   ├── __init__.py
│   ├── base.py                   # Base solver class
│   ├── BOAnBC_vao2_eetc.py       # Branch Outer Approximation and Benders Cuts solver
│   ├── BnBC_vao2_eetc.py         # Branch and Benders Cuts solver
│   ├── BnOA_vao2.py              # Branch and Outer Approximation solver
│   ├── BD_vao2_eetc.py           # Benders Decomposition solver
│   ├── Int_vao2_eetc.py          # Integrated solver
│   ├── Seq_vao2_eetc.py          # Sequential solver
│   ├── SeqBnOA_vao2_eetc.py      # Sequential Branch and Outer Approximation solver
│   └── greedy_vao.py             # Greedy heuristic for VAO
│
├── utils/                        # Utility functions
│   ├── __init__.py
│   ├── ground_gen.py             # Ground profile generation
│   ├── gurobi.py                 # Gurobi helper functions
│   └── plotting.py               # Plotting utilities
│
├── plot_traction_brake_curve.py  # Script for plotting train force curves
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. Clone this repository:

```bash
git clone <repository-url>
cd BOAnBC_github
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Note:** This project requires Gurobi Optimizer. Please ensure you have a valid Gurobi license installed.

## Requirements

- Python 3.8+
- numpy
- matplotlib
- gurobipy
- pandas
- PyQt5

## Usage

### Running the BOAnBC Solver

```python
from configs import GroundConfigM2, TrainConfig
from solvers import BOAnBC_vao2_eetc
from solvers.base import build_train_instances

# Create ground configuration
gcfg = GroundConfigM2("HP3")

# Create train instances
train_instances = build_train_instances(gcfg, ...)

# Run solver
solver = BOAnBC_vao2_eetc.solve_vao_eetc_boanbc(
    gcfg,
    train_instances,
    mip_gap=0.01,
    time_limit=3600,
    verbose=True
)
```

### Plotting Train Force Curves

```bash
python plot_traction_brake_curve.py
```

## Key Features

- **BOAnBC Algorithm**: Hybrid decomposition combining Branch-and-Bound, Outer Approximation, and Benders Decomposition
- **Energy-Efficient Train Control**: SOCP-based train control optimization
- **Vertical Alignment Optimization**: Mixed-integer quadratic programming with SOC constraints
- **Multiple Solution Algorithms**: Implementation of various decomposition methods for comparison
- **Visualization Tools**: Built-in plotting utilities for results analysis

## Contact

### Yichen Sun

- Email: <yichensun@my.swjtu.edu.cn>
- Email: <sun_yichen@foxmail.com>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
[Citation information to be added]
```
