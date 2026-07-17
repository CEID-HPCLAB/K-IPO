# ⚖️ K-IPO: Kendall-constrained Importance Preserving Oversampling for Imbalanced Tabular Data
[![GitHub release](https://img.shields.io/github/v/release/CEID-HPCLAB/K-IPO?color=%238FD9FB)](https://github.com/CEID-HPCLAB/K-IPO/releases)
[![License](https://img.shields.io/badge/License-Apache--2.0-FFDEAD)](https://www.apache.org/licenses/LICENSE-2.0)  <br>

[![Python 3.10](https://img.shields.io/badge/Python-3.10-purple?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/) 
[![XAI](https://img.shields.io/badge/XAI-Explainable%20AI-228B22)](#)
![Oversampling](https://img.shields.io/badge/Oversampling-001594?style=flat&logo=dna&logoColor=white)

**K-IPO** is a generator-agnostic, *generate-then-select* framework for imbalanced tabular data classification that preserves the original feature importance ranking during data augmentation. K-IPO iteratively generates minority-class candidates and accepts them only if their inclusion maintains a user-defined minimum Kendall’s tau ($\tau$) correlation with the reference feature importance ranking. Optionally, stricter constraints can be enforced on the highest-ranked top-k features. Evaluated on 20 imbalanced binary classification datasets using three classifiers and multiple explanation methods, K-IPO achieved the best or tied-best results in feature importance preservation, explanation consistency, and class separability compared with existing oversampling methods, including both conventional and generative approaches. It also generally improved predictive performance while maintaining competitive computational overhead.


## Table of Contents
- [Prerequisites & Installation](#prerequisites--installation)
- [Usage](#usage)
    - [Demo Example (API)](#demo-example-api)
    - [End-to-End Pipeline for AI4I2020](#end-to-end-pipeline-for-ai4i2020)
- [Datasets](#datasets)
    - [YAML Configuration Files](#yaml-configuration-files)
- [Performance Evaluation](#performance-evaluation)
- [File Structure](#file-structure)
- [Future Directions](#future-directions)
- [Acknowledgments](#acknowledgments)

## Prerequisites & Installation

Before installing the project, make sure the following requirements are satisfied:

1. **Python** (**3.10 <= version < 3.11**) is installed on your system (experiments were conducted with Python **3.10.20**).

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
> The [PiML](https://selfexplainml.github.io/PiML-Toolbox/_build/html/install.html) toolbox requires `pandas (<2.0.0)`, `numpy (<1.24.0)`, and `scipy (==1.5.3)`. These outdated dependencies conflict with other packages required by K-IPO, and `pip` may fail to automatically resolve the resulting dependency conflicts. To avoid installation issues, a [`setup.sh`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/setup.sh) script is provided to manually install the required dependency versions and configure the K-IPO environment.

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

### YAML Configuration Files

Each dataset is paired with a corresponding `YAML` configuration file that provides K-IPO with the required information to perform the data augmentation process. A representative `YAML` configuration file is structured as follows:

```yaml
# Columns to be dropped before augmentation
drop_cols:
  - column 1
  - column 2

# Target variable configuration
target_col:
  name: name of the target variable
  encoding: True # Set to True if the target variable is already label encoded, False otherwise

# Names of numerical columns
num_cols:
  - column 1
  - column 2

# Names and optional ordering of categorical columns
cat_cols:
  - column 1:
      order: # Specify an order if a hierarchical pattern exists

# CSV separator (default: ',')
sep: ;
```

> [!IMPORTANT]
> The [AI4I2020](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) dataset and its `YAML` configuration file are bundled with the repository under the [`datasets/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/datasets) folder and can be used directly for the K-IPO [API demonstration](https://github.com/CEID-HPCLAB/K-IPO/blob/main/example.py) and [end-to-end performance evaluation workflow](https://github.com/CEID-HPCLAB/K-IPO/blob/main/demo.ipynb) without any additional setup.


## Performance Evaluation

The experimental evaluation presented in the manuscript consists of **three** stages: (a) selecting the most **suitable candidate data generator** to be used as the basis for K-IPO synthetic sample generation, (b) performing a **sensitivity analysis** across the 20 evaluated datasets to determine the optimal K-IPO configuration parameters, namely the Kendall's tau ($\tau$) threshold and the top-k ordering constraint, for each dataset, and (c) evaluating K-IPO against existing data generation and oversampling approaches in terms of **feature importance preservation**, **explanation consistency**, **class separability**, and **predictive performance**.

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

> [!IMPORTANT] 
> The execution of the above commands requires all 20 raw datasets to be available locally. Therefore, the datasets must be downloaded beforehand using the commands provided in the [Datasets](#datasets) section.


Using the [`heatmap.py`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/heatmap.py)script located in the [`experiments/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments)folder, you can generate the heatmaps presented in **Figure 3** of the manuscript. To generate the heatmaps, run:
```bash
python ./experiments/heatmap.py 
```

The [`evaluation.ipynb`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/evaluation.ipynb) notebook can be used to generate **Tables 5, 6**, and **9** (Friedman test results) and **Figure 5** (critical difference diagrams) reported in the manuscript.

> [!NOTE]
> The [`f-score_anova.py`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/f-score_anova.py) and [`perftime_analysis.py`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/perftime_analysis.py) scripts can be used to generate the Figures 2 and 4 of the manuscript, respectively. 


## File Structure

- [`datasets/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/datasets): Evaluated datasets (demo: [AI4I2020](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset))
    - [`config/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/datasets/config): `YAML` configuration files for the datasets
    - [`data/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/datasets/data): Raw imbalanced datasets
    - [`download.sh`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/datasets/download.sh):  Script for downloading the datasets and their corresponding `YAML` configuration files
- [`experiments/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments): Experiments folder
    - [`evaluation/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/evaluation): Augmented datasets and evaluation results per dataset (in total: 20)
        - [`abalone/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/evaluation/abalone): Results for abalone dataset
            - [`datasets/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/evaluation/abalone/datasets): Augmented datasets 
                - [`CTGAN/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/evaluation/abalone/datasets/CTGAN): Datasets generated with CTGAN (in total: 10)
                - [`GaussianCopula/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/evaluation/abalone/datasets/GaussianCopula): Datasets generated with GaussianCopula (in total: 10)
                - [`K-IPO/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/evaluation/abalone/datasets/K-IPO): Datasets generated with K-IPO (in total: 10)
                - [`SMOTENC/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/evaluation/abalone/datasets/SMOTENC): Datasets generated with SMOTENC (in total: 10)
                - [`TabDDPM/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/evaluation/abalone/datasets/TabDDPM): Datasets generated with TabDDPM (in total: 10)
                - [`TVAE/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/evaluation/abalone/datasets/TVAE): Datasets generated with TVAE (in total: 10)
            - [`importance/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/evaluation/abalone/importance): Feature importance rankings
                - [`CTGAN/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/evaluation/abalone/importance/CTGAN): Feature importance rankings for datasets generated with CTGAN (in total: 10)
                - [`GaussianCopula/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/evaluation/abalone/importance/GaussianCopula): Feature importance rankings for datasets generated with GaussianCopula (in total: 10)
                - [`K-IPO/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/evaluation/abalone/importance/K-IPO): Feature importance rankings for datasets generated with K-IPO (in total: 10)
                - [`SMOTENC/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/evaluation/abalone/importance/SMOTENC): Feature importance rankings for datasets generated with SMOTENC (in total: 10)
                - [`TabDDPM/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/evaluation/abalone/importance/TabDDPM): Feature importance rankings for datasets generated with TabDDPM (in total: 10)
                - [`TVAE/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/evaluation/abalone/importance/TVAE): Feature importance rankings for datasets generated with TVAE (in total: 10)
            - [`results/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/evaluation/abalone/results): Evaluation results 
                - [`CTGAN.csv`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/evaluation/abalone/results/CTGAN.csv): Detailed results for the 10 CTGAN augmented datasets
                - [`GaussianCopula.csv`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/evaluation/abalone/results/GaussianCopula.csv): Detailed results for the 10 GaussianCopula augmented datasets
                - [`K-IPO.csv`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/evaluation/abalone/results/K-IPO.csv): Detailed results for the 10 K-IPO augmented datasets
                - [`SMOTENC.csv`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/evaluation/abalone/results/SMOTENC.csv): Detailed results for the 10 SMOTENC augmented datasets
                - [`TabDDPM.csv`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/evaluation/abalone/results/TabDDPM.csv): Detailed results for the 10 TabDDPM augmented datasets
                - [`TVAE.csv`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/evaluation/abalone/results/TVAE.csv): Detailed results for the 10 TVAE augmented datasets
        - *(The same pattern is repeated for each of the 20 datasets)*
    - [`external/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/external): External figures included in the manuscript
        - [`pdf/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/external/pdf): .pdf format
            - [`f-score_anova.pdf`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/external/pdf/f-score_anova.pdf): Figure 2 from the manuscript
            - [`runtime.pdf`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/external/pdf/runtime.pdf): Figure 4 from the manuscript
        - [`png/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/external/png): .png format, DPI: 1200
            - [`f-score_anova.png`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/external/png/f-score_anova.png): Figure 2 from the manuscript
            - [`runtime.png`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/external/png/runtime.png): Figure 4 from the manuscript
    - [`performance_analysis/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/performance_analysis): Reported execution times of the evaluated generators for data generation
        - [`abalone/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/performance_analysis/abalone): Runtime results for abalone dataset
            - [`CTGAN.csv`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/performance_analysis/abalone/CTGAN.csv): Detailed runtime results for CTGAN
            - [`GaussianCopula.csv`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/performance_analysis/abalone/GaussianCopula.csv): Detailed runtime results for CTGAN
            - [`K-IPO.csv`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/performance_analysis/abalone/K-IPO.csv): Detailed runtime results for K-IPO
            - [`SMOTENC.csv`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/performance_analysis/abalone/SMOTENC.csv): Detailed runtime results for SMOTENC
            - [`TabDDPM.csv`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/performance_analysis/abalone/TabDDPM.csv): Detailed runtime results for TabDDPM
            - [`TVAE.csv`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/performance_analysis/abalone/TVAE.csv): Detailed runtime results for TVAE
        - *(The same pattern is repeated for each of the 20 datasets)*
    - [`selection/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/selection): Augmented datasets and evaluation results for selecting the K-IPO backbone generator (in total: 20)
        - [`abalone/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/selection/abalone): Results for abalone dataset
            - [`datasets/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/selection/abalone/datasets): Augmented datasets 
                - [`CTGAN/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/selection/abalone/datasets/CTGAN): Datasets generated with CTGAN (in total: 10)
                - [`GaussianCopula/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/selection/abalone/datasets/GaussianCopula): Datasets generated with GaussianCopula (in total: 10)
                - [`SMOTENC/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/selection/abalone/datasets/SMOTENC): Datasets generated with SMOTENC (in total: 10)
                - `TabDDPM/`: Datasets generated with TabDDPM (in total: 10) (for the `abalone` dataset, TabDDPM failed to satisfy the required Kendall's $\tau$ threshold of 0.7)
                - [`TVAE/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/selection/abalone/datasets/TVAE): Datasets generated with TVAE (in total: 10)
            - [`results/`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/selection/abalone/results): Evaluation results
                - [`CTGAN.csv`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/selection/abalone/results/CTGAN.csv): Detailed results for the ten CTGAN augmented datasets
                - [`GaussianCopula.csv`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/selection/abalone/results/GaussianCopula.csv): Detailed results for the ten GaussianCopula augmented datasets
                - [`SMOTENC.csv`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/selection/abalone/results/SMOTENC.csv): Detailed results for the ten SMOTENC augmented datasets
                - `TabDDPM.csv`: Detailed runtime results for the ten TabDDPM augmented datasets (for the `abalone` dataset, TabDDPM failed to satisfy the required Kendall's $\tau$ threshold of 0.7)
                - [`TVAE.csv`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/selection/abalone/results/TVAE.csv): Detailed results for the ten TVAE augmented datasets
        - *(The same pattern is repeated for each of the 20 datasets)*
    - [`sensitivity_analysis/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/sensitivity_analysis): Sensitivity analysis of K-IPO on the evaluated datasets
        - [`datasets/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/sensitivity_analysis/datasets): Augmented datasets 
            - [`abalone/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/sensitivity_analysis/datasets/abalone): Abalone augmented datasets (in total: 12)
            - *(The same pattern is repeated for each of the 20 datasets)*
        - [`heatmaps/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/sensitivity_analysis/heatmaps): Heatmaps of generated datasets (Figure 3 from the manuscript)
            - [`pdf/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/sensitivity_analysis/heatmaps/pdf): Heatmaps of generated datasets (.pdf format)
                - [`abalone.pdf`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/sensitivity_analysis/heatmaps/pdf/abalone.pdf): Heatmap for the abalone dataset (.pdf format)
                - *(The same pattern is repeated for each of the 20 datasets)*
            - [`png/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/sensitivity_analysis/heatmaps/png): Heatmaps of generated datasets (.png format, DPI: 1200)
                - [`abalone.png`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/sensitivity_analysis/heatmaps/png/abalone.png): Heatmap for the abalone dataset (.png format)
                - *(The same pattern is repeated for each of the 20 datasets)*
        - [`results/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/experiments/sensitivity_analysis/results): Evaluation results from the sensitivity analysis
            - [`abalone.csv`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/sensitivity_analysis/results/abalone.csv): Detailed sensitivity analysis results for the abalone dataset
            - *(The same pattern is repeated for each of the 20 datasets)*
    - [`config.yml`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/config.yml): Experiments configuration file
    - [`eval.py`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/eval.py): Evaluation of augmented datasets (both predictive capabilites and top-K overlap)
    - [`eval.sh`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/eval.sh): Script for running the three stages of the experimental evaluation
    - [`evaluation.ipynb`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/evaluation.ipynb): Notebook for generating Tables 5, 6, and 9 and Figure 5 reported in the manuscript
    - [`f-score_anova.py`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/f-score_anova.py): Generates Figure 2 from the manuscript
    - [`generator.py`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/generator.py): Generation of augmented datasets
    - [`heatmap.py`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/heatmap.py): Generation of heatmaps (Figure 5 from the manuscript)
    - [`imbalance.py`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/imbalance.py): Converts a balanced dataset into an imbalanced one
    - [`perftime_analysis.py`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/perftime_analysis.py): Generates Figure 4 from the manuscript
    - [`topk_features.yml`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/topk_features.yml): Number of top features per dataset (>= 90% cumulative ANOVA F-score)
    - [`utils.py`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/experiments/utils.py): Helper functions for running experiments
- [`ground_truth/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/ground_truth): Ground truth for the evaluated datasets
  -  [`data/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/ground_truth/data): Feature importance rankings for the evaluated datasets
  -  [`extract.py`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/ground_truth/extract.py): Ground truth computation using the seven supported methods
  -  [`extract.sh`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/ground_truth/extract.sh): Script for generating the ground-truth feature importance rankings for the 20 evaluated datasets
  -  [`gini.py`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/ground_truth/gini.py): Script for computing the Gini index (ACK: [scikit-feature](https://github.com/jundongl/scikit-feature))
  -  [`laplacian.py`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/ground_truth/laplacian.py): Script for computing Laplacian Scores (ACK: [scikit-feature](https://github.com/jundongl/scikit-feature))
  -  [`utils.py`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/ground_truth/utils.py): Helper functions for ground truth extraction
- [`src/kipo/`](https://github.com/CEID-HPCLAB/K-IPO/tree/main/src/kipo): Core implementation of the K-IPO package
  -  [`generator.py`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/src/kipo/generator.py): Generation of augemented datasets using the supported generators
  -  [`importance.py`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/src/kipo/importance.py): Feature importance computation using the three supported XAI methods
  -  [`models.py`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/src/kipo/models.py): Definition of models used for evaluation in the experiments
  -  [`selector.py`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/src/kipo/selector.py): Core backbone class of K-IPO
  - [`sofi.py`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/src/kipo/sofi.py): Source code of [Sparseness-Optimized Feature Importance (SOFI)](https://github.com/gnapoles/sofi-explainer) explainer
  - [`utils.py`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/src/kipo/utils.py): Helper functions
- [`config.yml`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/config.yml): Configuration file for demo example
- [`demo.ipynb`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/demo.ipynb): End-to-end pipeline for data generation and evaluation on the [AI4I2020](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) predictive maintenance dataset
- [`example.py`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/example.py): K-IPO API demonstration
- [`pyproject.toml`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/pyproject.toml): Project configuration file containing package metadata and dependency specifications
- [`requirements.txt`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/requirements.txt): Python dependencies
- [`setup.py`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/setup.py): Deprecated setup script for the K-IPO package
- [`setup.sh`](https://github.com/CEID-HPCLAB/K-IPO/blob/main/setup.sh): Script for setting up the K-IPO package environment and installing the required dependencies

## Future Directions
- [ ] Improved API documentation
- [ ] Support for large-scale datasets through a distributed K-IPO implementation on multi-node clusters
- [ ] Integration of uncertainty-aware rank preservation constraints
- [ ] Support for multi-class and multi-label classification
- [ ] Extend K-IPO beyond tabular data (e.g., image data)
- [ ] Automatic selection of rank preservation constraints

## Acknowledgments

This research was funded by the "ARCHIMEDES Unit: Research in Artificial Intelligence, Data Science, and Algorithms" (MIS 5154714) under Greece's National Recovery and Resilience Plan, funded by the European Union – NextGenerationEU.


[open-mpi-link]: https://www.open-mpi.org/
[mpich-link]: https://www.mpich.org/