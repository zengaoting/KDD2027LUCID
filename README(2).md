# KDD2026LUCID

This is the official repo for KDD 2026 Research Track paper ''**Interpretable Unsupervised Community Detection with LLM-Symbolized Structured Processes**''. 


If you find this work useful in your method, you can cite the paper as below:
```latex

```

## Overview

We propose LUCID, an LLM-guided interpretable, training-free and Unsupervised Community Detection method. Inspired by phase-transition kinetics in natural systems, LUCID is designed as a four-stage pipeline:

1. **Local-View Community Initialization:** Decomposes the graph into localized k-ego networks and assigns unsupervised node role labels to capture structural semantics.
2. **Multi-factor Community Merge:** Utilizes LLM-induced decision trees to evaluate structural compatibility and iteratively merge local communities using dynamic orchestration.
3. **Multi-grain Community Refinement:** Applies LLM-generated coarse-to-fine rules in parallel to refine node memberships and eliminate boundary noise.
4. **Global-view Community Selection:** Identifies high-quality communities using a structure-stability-regularized metric that balances internal cohesion and boundary clarity.


## Run the Codes

### Repo Introduction
This repo contains the following contents:
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
    │   └── facebook-1.90_dataset_description.txt  
    ├── amazon/
    │   └── amazon-1.90_dataset_description.txt    
    ├── dblp/
    │   └── dblp-1.90_dataset_description.txt      
    ├── lj/
    │   └── lj-1.90_dataset_description.txt       
    └── twitter/
        └── twitter-1.90_dataset_description.txt    
```

### Environmental Requirements

1. Python 3.7 or above
2. Pytorch
3. Pytorch-Geometric (PyG)
4. OpenAI
5. Other dependencies: numpy, scipy, scikit-learn, tqdm, crowdkit, joblib

Install dependencies:
```bash
pip install torch torch-geometric numpy scipy scikit-learn tqdm openai crowdkit joblib
```

### Running LUCID

Set your OpenAI API credentials:
```bash
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_MODEL_NAME="gpt-5.2"  # or your preferred model
```

Execute `run.py` for the overall LUCID pipeline:
```bash
python run.py --dataset=DATASET_NAME --device=DEVICE
```

Descriptions of arguments (for more options, please refer to `run.py`)
```bash
--dataset [facebook, amazon, dblp, twitter, lj]: the dataset to run 
--k: number of hops for ego network, default as 2
--merge_sample_size: number of samples for merge rule generation, default as 3
--refine_sample_size: number of samples for refinement rule generation, default as 10
--tree_depth_range: depth range for decision tree, default as "11 13"
--epsilon: penalty coefficient for RDC metric, default as 0.2
--top_b: neighbors to consider for merging, default as 3
```

## Performance 
 | Metric | Dataset | BigClam | ComE | CommGAN | Bespoke | SEAL | CLARE | PROCOM | LUCID (Ours) | Improv. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |:-------------| :--- |
| **F1 Score** | Facebook | 36.50 | <u>37.78</u> | 27.24 | 32.82±0.90 | 28.86±14.66 | 26.76±2.44 | 36.97±1.48 | **40.95**    | +8.4% |
| | Amazon | 71.39 | 78.51 | 69.52 | 64.38±1.56 | <u>84.00±4.07</u> | 80.61±1.92 | 83.97±0.59 | **91.71**    | +9.2% |
| | LiveJournal| 40.69 | N/A | 41.52 | 38.37±1.43 | 52.85±5.79 | 46.32±3.81 | <u>53.82±2.52</u> | **56.57**    | +5.1% |
| | DBLP | 40.46 | N/A | N/A | 45.65±1.64 | 26.75±14.00 | <u>52.15±1.36</u> | 50.38±2.56 | **55.03**    | +5.5% |
| | Twitter | 29.60 | 25.94 | N/A | <u>29.78±0.58</u> | 15.33±2.60 | 18.69±1.17 | 26.99±1.79 | **33.03**    | +10.9% |
| **Jaccard** | Facebook | 26.08 | <u>27.84</u> | 19.13 | 23.71±0.99 | 20.21±11.13 | 18.19±1.77 | 26.66±1.16 | **30.87**    | +10.9% |
| | Amazon | 61.38 | 70.07 | 61.47 | 58.14±1.66 | 74.00±5.41 | 73.06±2.87 | <u>75.47±0.63</u> | **87.03**    | +15.3% |
| | LiveJournal| 31.72 | N/A | 35.52 | 32.43±1.28 | 42.34±5.58 | 37.31±3.47 | <u>44.32±2.51</u> | **48.99**    | +10.5% |
| | DBLP | 28.78 | N/A | N/A | 40.33±1.66 | 18.75±11.69 | <u>40.40±1.21</u> | 39.12±2.59 | **45.74**    | +13.2% |
| | Twitter | 18.75 | 16.41 | N/A | <u>20.15±0.42</u> | 8.89±1.81 | 11.51±0.79 | 17.85±1.23 | **22.16**    | +10.0% |