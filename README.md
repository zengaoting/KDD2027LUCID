# LUCID: LLM-Based Community Detection

This is the official repository for the KDD 2026 paper **"LUCID: LLM-Based Community Detection"**.

> **Note**: This repository contains the implementation of LUCID, a Large Language Model (LLM)-based approach for community detection in graphs that leverages LLMs to generate interpretable decision rules.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Data Format](#data-format)
- [Usage](#usage)
- [Performance](#performance)
- [File Structure](#file-structure)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Overview

LUCID leverages LLMs to induce interpretable decision rules for community detection through four main stages:

1. **Stage 1: Initialize** - Extract k-hop ego networks for each node and assign node labels
2. **Stage 2: Merge** - Use LLM-generated decision trees to merge similar k-ego networks
3. **Stage 3: Refine** - Apply LLM-generated functions to refine community boundaries
4. **Stage 4: Rank** - Rank communities using quality metrics (Density-Weighted Conductance)

### Key Features

- **Interpretable**: LLM-generated decision rules provide transparency in community detection
- **Adaptive**: Automatically adapts to different graph structures through LLM-based rule generation
- **Effective**: Achieves competitive performance on standard benchmark datasets

## Requirements

### Python Packages

- Python 3.7+
- torch
- torch-geometric
- numpy
- scipy
- scikit-learn
- tqdm
- openai
- crowdkit
- joblib

Install dependencies:
```bash
pip install torch torch-geometric numpy scipy scikit-learn tqdm openai crowdkit joblib
```

### Environment Variables

Set your OpenAI API credentials:
```bash
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_MODEL_NAME="gpt-5.2"  # or your preferred model
```

## Data Format

Place your datasets in the `data/` directory with the following structure:
```
data/
  {dataset_name}/
    {dataset_name}-1.90.ungraph.txt              # Edge list (required)
    {dataset_name}-1.90.cmty.txt                 # Ground truth communities (required)
    {dataset_name}-1.90_dataset_description.txt   # Dataset description (required for LLM)
    {k}-ego.txt                                  # Pre-computed k-ego networks (optional, auto-generated if missing)
```

### Required Files

Each dataset directory must contain:

1. **`{dataset_name}-1.90.ungraph.txt`**: Edge list file
   - Format: One edge per line, space-separated node IDs
   - Example: `0 1\n1 2\n2 3\n`

2. **`{dataset_name}-1.90.cmty.txt`**: Ground truth communities
   - Format: One community per line, space-separated node IDs
   - Example: `0 1 2\n3 4 5\n`

3. **`{dataset_name}-1.90_dataset_description.txt`**: Dataset description for LLM context
   - This file provides global information about the dataset to help LLM generate better rules
   - Should include: dataset overview, scale (nodes, edges, communities), community size statistics
   - Example format:
     ```
     数据集概述
     ----------
     [Dataset description]
     
     数据集规模
     ----------
     - 节点总数：X个节点
     - 边总数：Y条无向边
     - 社区数量：Z个标注社区
     - 社区规模统计：...
     ```

### Optional Files

- **`{k}-ego.txt`**: Pre-computed k-ego networks (where k is the number of hops)
  - If this file exists, the code will load it instead of recomputing
  - Format: One line per node, starting with node ID followed by its k-ego network nodes
  - Example: `0 1 2 3\n1 0 2 4\n`
  - If missing, k-ego networks will be computed automatically and saved to this file

### Dataset Description Files

The dataset description files (`*_dataset_description.txt`) are **critical** for the LLM-based approach as they provide global context about the dataset structure. These files are included in the repository for all supported datasets:
- `facebook-1.90_dataset_description.txt`
- `amazon-1.90_dataset_description.txt`
- `dblp-1.90_dataset_description.txt`
- `lj-1.90_dataset_description.txt`
- `twitter-1.90_dataset_description.txt`

**Note**: The actual graph data files (`.ungraph.txt`, `.cmty.txt`) are not included in the repository due to size constraints. You need to download them separately from the [SNAP dataset repository](http://snap.stanford.edu/data/index.html) or provide your own data following the format above. See [DATA_README.md](DATA_README.md) for detailed instructions.

## Usage

### Basic Usage

```bash
python run.py --dataset facebook --device cuda:0
```

### Command Line Arguments

#### Required Arguments
- `--dataset`: Dataset name. Options: `['amazon', 'dblp', 'lj', 'facebook', 'twitter']`
- `--device`: Device to use. Options: `cuda:0`, `cuda:1`, `cpu`, etc.

#### Optional Arguments

**K-EGO Parameters:**
- `--k`: Number of hops for k-ego network extraction (default: `2`)
- `--n_roles`: Number of node labels for k-means clustering (default: `4`)

**LLM Sampling Parameters:**
- `--merge_sample_size`: Number of k-ego samples for LLM merge (default: `3`)
- `--refine_sample_size`: Number of communities for refinement learning (default: `10`)
- `--refine_jobs`: Number of parallel processes for refinement (default: `4`)

**Input Format:**
- `--input_format`: Input format for LLM rules. Options: `['degree', 'adjacency_list']` (default: `degree`)

**Neighbor Sampling:**
- `--max_neighbors_per_center`: Maximum neighbors per center node. Use `-1` for all neighbors (default: `3`)
- `--topk_neighbors_merge`: Top-k similar neighbors for merge. Use `-1` for all neighbors (default: `3`)

**Ranking:**
- `--ranking_method`: Ranking method. Options: `['conductance']` (default: `conductance`)
- `--epsilon`: Epsilon parameter for Density-Weighted Conductance (default: `0.3`)
- `--tree_depth_range`: Range of decision tree depth [MIN MAX] (default: `8 12`)

**Data Path:**
- `--data_root`: Root directory for data files (default: `./data`)

### Example Commands

**Facebook dataset with default settings:**
```bash
python run.py --dataset facebook --device cuda:0
```

**Amazon dataset with custom parameters:**
```bash
python run.py \
    --dataset amazon \
    --device cuda:0 \
    --k 2 \
    --input_format adjacency_list \
    --merge_sample_size 3 \
    --refine_sample_size 10 \
    --ranking_method conductance \
    --epsilon 0.2 \
    --tree_depth_range 11 13
```

## Performance

We conduct comprehensive experiments on five benchmark datasets from the SNAP repository. The following table reports the overall performance comparison with baseline methods.

### Overall Performance

| Dataset   | Metric    | BigClam | ComE     | CommunityGAN | Bespoke        | SEAL           | CLARE          | LUCID (Ours)   |
|-----------|-----------|---------|----------|--------------|----------------|----------------|----------------|----------------|
| Facebook  | F1-Score  | 32.92   | 27.92    | 32.05        | 29.67±0.85     | 31.10±3.84     | 28.53±1.36     | **TBD**        |
|           | Precision | -       | -        | -            | -              | -              | -              | **TBD**        |
|           | Recall    | -       | -        | -            | -              | -              | -              | **TBD**        |
|           | NMI       | -       | -        | -            | -              | -              | -              | **TBD**        |
| Amazon    | F1-Score  | 53.79   | 48.23    | 51.09        | 80.38±0.64     | 82.26±4.04     | 78.89±2.10     | **TBD**        |
|           | Precision | -       | -        | -            | -              | -              | -              | **TBD**        |
|           | Recall    | -       | -        | -            | -              | -              | -              | **TBD**        |
|           | NMI       | -       | -        | -            | -              | -              | -              | **TBD**        |
| LiveJournal| F1-Score | 39.17   | N/A      | 40.67        | 30.98±1.55     | 42.85±2.60     | 45.38±4.07     | **TBD**        |
|           | Precision | -       | -        | -            | -              | -              | -              | **TBD**        |
|           | Recall    | -       | -        | -            | -              | -              | -              | **TBD**        |
|           | NMI       | -       | -        | -            | -              | -              | -              | **TBD**        |
| DBLP      | F1-Score  | 40.41   | 25.24    | N/A          | 41.55±0.40     | 41.74±6.35     | 48.75±2.51     | **TBD**        |
|           | Precision | -       | -        | -            | -              | -              | -              | **TBD**        |
|           | Recall    | -       | -        | -            | -              | -              | -              | **TBD**        |
|           | NMI       | -       | -        | -            | -              | -              | -              | **TBD**        |
| Twitter   | F1-Score  | 24.33   | 15.89    | N/A          | 29.85±0.15     | 16.97±1.32     | 20.05±0.88     | **TBD**        |
|           | Precision | -       | -        | -            | -              | -              | -              | **TBD**        |
|           | Recall    | -       | -        | -            | -              | -              | -              | **TBD**        |
|           | NMI       | -       | -        | -            | -              | -              | -              | **TBD**        |

> **Note**: Results marked as "TBD" (To Be Determined) will be updated with actual experimental results. The baseline results for traditional methods (BigClam, ComE, CommunityGAN) are reported from SEAL and CLARE papers. For semi-supervised methods (Bespoke, SEAL, CLARE), results are obtained by running their released codes under the same experimental settings.

### Experimental Setup

- **Datasets**: We evaluate on five datasets from SNAP: Facebook, Amazon, LiveJournal, DBLP, and Twitter
- **Evaluation Metrics**: F1-Score, Precision, Recall, NMI (Normalized Mutual Information), Jaccard Similarity
- **Baselines**: We compare with traditional methods (BigClam, ComE, CommunityGAN) and semi-supervised methods (Bespoke, SEAL, CLARE)
- **Reproducibility**: All experiments are run with fixed random seeds, and results are reported as mean ± standard deviation over multiple runs

## File Structure

```
KDD2026LUCID/
├── run.py                    # Main execution script
├── utils.py                  # Utility functions (node labeling, file I/O)
├── snap_datasets.py          # Dataset loading functions
├── evaluate.py               # Evaluation metrics
├── README.md                 # This file
├── DATA_README.md            # Detailed data file documentation
├── .gitignore                # Git ignore rules
└── data/                     # Data directory
    ├── facebook/
    │   └── facebook-1.90_dataset_description.txt  ✓ Included
    ├── amazon/
    │   └── amazon-1.90_dataset_description.txt    ✓ Included
    ├── dblp/
    │   └── dblp-1.90_dataset_description.txt       ✓ Included
    ├── lj/
    │   └── lj-1.90_dataset_description.txt         ✓ Included
    └── twitter/
        └── twitter-1.90_dataset_description.txt     ✓ Included
```

**Note**: The dataset description files (`*_dataset_description.txt`) are **critical** for the LLM-based approach and are included in the repository. However, the actual graph data files (`.ungraph.txt`, `.cmty.txt`) are not included due to size constraints. See [DATA_README.md](DATA_README.md) for details on how to obtain and prepare the data files.

## Output

The script outputs:
- Community detection results with evaluation metrics (Precision, Recall, F1, Jaccard, NMI)
- Predicted communities saved to `output_comm/` directory
- Ranking scores for each community

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{lucid2026,
  title={LUCID: LLM-Based Community Detection},
  author={...},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2026}
}
```

## License

[Add your license information here]

## Contact

[Add contact information here]
