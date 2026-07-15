# ⚖️ K-IPO: Kendall-constrained Importance Preserving Oversampling for Imbalanced Tabular Data
[![GitHub release](https://img.shields.io/github/v/release/CEID-HPCLAB/Mneme?include_prereleases&color=%238FD9FB)](https://github.com/CEID-HPCLAB/Mneme/releases)
[![License](https://img.shields.io/badge/License-Apache--2.0-FFDEAD)](https://www.apache.org/licenses/LICENSE-2.0)  <br>

[![Python 3.10](https://img.shields.io/badge/Python-3.10-purple?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/) 
![ΧΑΙ](https://img.shields.io/badge/ΧΑΙ-228B22?style=flat&logo=dna&logoColor=white)
![Oversampling](https://img.shields.io/badge/Oversampling-001594?style=flat&logo=dna&logoColor=white)

**K-IPO** is a generator-agnostic, *generate-then-select* framework for imbalanced tabular data classification that preserves the original feature importance ranking during data augmentation. K-IPO iteratively generates minority-class candidates and accepts them only if their inclusion maintains a user-defined minimum Kendall’s tau ($\tau$) correlation with the reference feature importance ranking. Optionally, stricter constraints can be enforced on the highest-ranked (top-$k$) features. Evaluated on 20 imbalanced binary classification datasets using three classifiers and multiple explanation methods, K-IPO achieved the best or tied-best results in feature importance preservation, explanation consistency, and class separability compared with existing oversampling methods, including both conventional and generative approaches. It also generally improved predictive performance while maintaining competitive computational overhead.


## Table of Contents
- [Prerequisites & Installation](#prerequisites--installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Performance Evaluation](#performance-evaluation)
- [Reproducibility](#reproducibility)
- [File Structure](#file-structure)
- [Acknowledgments](#acknowledgments)

## Prerequisites & Installation

Before installing the project, make sure the following requirements are satisfied:

1. **Python** (version >= **3.10**) is installed on your system (experiments were conducted with Python **3.10.20**). 

2. An **MPI** implementation is installed on your system, such as [OpenMPI][open-mpi-link] (recommended) or [MPICH][mpich-link].

> [!WARNING]  
> Make sure that the `mpirun` command is available in your system's `PATH`, as it is required to launch MPI processes.

Clone the repository:
```bash
git clone https://github.com/CEID-HPCLAB/K-IPO.git
cd K-IPO
```

Install the **K-IPO** package and external dependencies:
```bash
# (Optional) Create and activate a Python virtual environment using Python 3.10
python3.10 -m venv venv
source venv/bin/activate # POSIX (bash/zsh)

chmod +x ./setup.sh
./setup.sh 
```

> [!WARNING] 
> The [PiML](https://selfexplainml.github.io/PiML-Toolbox/_build/html/install.html) toolbox requires `pandas (<2.0.0)`, `numpy (<1.24.0)`, and `scipy (==1.5.3)`. These outdated dependencies conflict with other packages required by K-IPO, and `pip` may fail to automatically resolve the required environment. To avoid installation issues, K-IPO provides a [setup.sh](https://github.com/CEID-HPCLAB/K-IPO/blob/main/setup.sh) script that manually installs the required dependencies and configures the environment.


## Acknowledgments
This research was funded by the "ARCHIMEDES Unit: Research in Artificial Intelligence, Data Science, and Algorithms" (MIS 5154714) under Greece's National Recovery and Resilience Plan, funded by the European Union – NextGenerationEU.



[open-mpi-link]: https://www.open-mpi.org/
[mpich-link]: https://www.mpich.org/