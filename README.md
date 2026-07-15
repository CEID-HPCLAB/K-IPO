# ⚖️ K-IPO: Kendall-constrained Importance Preserving Oversampling for Imbalanced Tabular Data
[![GitHub release](https://img.shields.io/github/v/release/CEID-HPCLAB/Mneme?include_prereleases&color=%238FD9FB)](https://github.com/CEID-HPCLAB/Mneme/releases)
[![License](https://img.shields.io/badge/License-Apache--2.0-FFDEAD)](https://www.apache.org/licenses/LICENSE-2.0)  <br>

[![Python 3.10](https://img.shields.io/badge/Python-3.10-purple?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/) 
[![XAI](https://img.shields.io/badge/XAI-Explainable%20AI-228B22)](#)
![Oversampling](https://img.shields.io/badge/Oversampling-001594?style=flat&logo=dna&logoColor=white)

**K-IPO** is a generator-agnostic, *generate-then-select* framework for imbalanced tabular data classification that preserves the original feature importance ranking during data augmentation. K-IPO iteratively generates minority-class candidates and accepts them only if their inclusion maintains a user-defined minimum Kendall’s tau ($\tau$) correlation with the reference feature importance ranking. Optionally, stricter constraints can be enforced on the highest-ranked (top-$k$) features. Evaluated on 20 imbalanced binary classification datasets using three classifiers and multiple explanation methods, K-IPO achieved the best or tied-best results in feature importance preservation, explanation consistency, and class separability compared with existing oversampling methods, including both conventional and generative approaches. It also generally improved predictive performance while maintaining competitive computational overhead.


## Table of Contents
- [Prerequisites & Installation](#prerequisites--installation)
- [Usage](#usage)
    - [Demo Example (API)](#demo-example-api)
    - [End-to-End Pipeline for AI4I2020](#end-to-end-pipeline-for-ai4i2020)
- [Datasets](#datasets)
- [File Structure](#file-structure)
- [Performance Evaluation](#performance-evaluation)
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
# (Recommended) Create and activate a Python virtual environment using Python 3.10
python3.10 -m venv venv
source venv/bin/activate # POSIX (bash/zsh)

chmod +x ./setup.sh
./setup.sh 
```

> [!WARNING] 
> The [PiML](https://selfexplainml.github.io/PiML-Toolbox/_build/html/install.html) toolbox requires `pandas (<2.0.0)`, `numpy (<1.24.0)`, and `scipy (==1.5.3)`. These outdated dependencies conflict with other packages required by K-IPO, and `pip` may fail to automatically resolve all dependency conflicts. To avoid installation issues, K-IPO provides a [`setup.sh`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/setup.sh) script that manually resolves these conflicts and configures the environment.

## Usage

### Demo Example (API)

The code listing below illustrates a representative example of how K-IPO can be applied to oversample an imbalanced tabular binary classification dataset. `TAU_THRESHOLD`, `TOPK_OVERLAP`, `TOPK_ORDERING`, and `BALANCE_RATIO` are configuration parameters that control the oversampling process. Their values are specified in a [YAML configuration file](https://github.com/CEID-HPCLAB/K-IPO/blob/main/config.yml), which, among other settings, defines the underlying generator used by K-IPO to generate new samples and the dataset on which the oversampling process is performed.

```python
from kipo.selector import KIPOSelector as KIPO

kipo = KIPO(num_features, tau_threshold = TAU_THRESHOLD, topk_ordering = TOPK_ORDERING, 
           topk_overlap = TOPK_OVERLAP)

kipo_X_aug, kipo_y_aug, info = kipo.select(X_train, y_train, X_test, y_test, 
                                           ratio = BALANCE_RATIO, 
                                           generator = gen_conf["method"], 
                                           preprocessing = pipeline, 
                                           **gen_conf["params"])

```

> [!NOTE]
> For a complete example demonstrating the K-IPO synthetic data generation workflow, we refer the reader to the [`example.py`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/example.py) script.

### End-to-End Pipeline for AI4I2020

The [`demo.ipynb`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/demo.ipynb) notebook presents an end-to-end pipeline for augmenting the [AI4I2020](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) predictive maintenance dataset. The pipeline evaluates and compares K-IPO against several state-of-the-art data generation and oversampling methods, including [CTGAN](https://github.com/sdv-dev/CTGAN), [TVAE](https://github.com/sdv-dev/CTGAN/blob/main/ctgan/synthesizers/tvae.py), [Gaussian Copula](https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/gaussiancopulasynthesizer) and [SMOTENC](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html). 

## Datasets

To download and install the datasets used in the manuscript, along with the corresponding `YAML` configuration files, run the following commands:
```bash
chmod +x ./datasets/download.sh
./datasets/download.sh 
```

> [!NOTE]
> The [AI4I2020](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) dataset and its `YAML` configuration file are bundled with the repository under the [`datasets/`](#https://github.com/CEID-HPCLAB/K-IPO/tree/main/datasets) folder and can be used directly for the K-IPO [API demonstration](https://github.com/CEID-HPCLAB/K-IPO/blob/main/example.py) and [end-to-end performance evaluation workflow](https://github.com/CEID-HPCLAB/K-IPO/blob/main/demo.ipynb) without any additional setup.


## File Structure

## Performance Evaluation

The experimental evaluation presented in the manuscript consists of **three** stages: (a) selecting the most **suitable candidate data generator** to be used as the basis for K-IPO synthetic sample generation, (b) performing a **sensitivity analysis** across the 20 evaluated datasets to determine the optimal K-IPO configuration parameters, namely the Kendall's tau ($\tau$) threshold and the top-$k$ ordering constraint, for each dataset, and (c) evaluating K-IPO against existing data generation and oversampling approaches in terms of **feature importance preservation**, **explanation consistency**, **class separability**, and **predictive performance**.

The results from all three evaluation stages can be reproduced using the [`eval.sh`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/eval.sh) script located in the experiments folder ([`experiments/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments)), as follows:
```bash
chmod +x ./experiments/eval.sh

# Run the base generator selection stage
./experiments/eval.sh A

# Run the K-IPO sensitivity analysis stage
./experiments/eval.sh B

# Run the K-IPO performance evaluation stage
./experiments/eval.sh C 
```

> [!WARNING] 
> The execution of the above commands requires all 20 raw datasets to be available locally. Therefore, the datasets must be downloaded beforehand using the commands provided in the [Datasets](#datasets) section.


Using the [`heatmap.py`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/heatmap.py)script located in the [`experiments/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments)folder, you can generate the heatmaps presented in **Figure 3** of the manuscript. To generate the heatmaps, run:
```bash
python ./experiments/heatmap.py 
```

The [`evaluation.ipynb`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/evaluation.ipynb) notebook can be used to generate **Tables 5, 6**, and **9** (Friedman test results) and **Figure 5** (critical diagrams) reported in the manuscript.

> [!NOTE]
> The [`f-score_anova.py`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/f-score_anova.py) and [`perftime_analysis.py`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/perftime_analysis.py) scripts can be used to generate the figures 2 and 4 of the manuscript, respectively. 


## Acknowledgments

This research was funded by the "ARCHIMEDES Unit: Research in Artificial Intelligence, Data Science, and Algorithms" (MIS 5154714) under Greece's National Recovery and Resilience Plan, funded by the European Union – NextGenerationEU.



[open-mpi-link]: https://www.open-mpi.org/
[mpich-link]: https://www.mpich.org/