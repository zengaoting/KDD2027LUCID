# Data Directory Structure

This directory contains dataset description files and serves as the data root for the LUCID code.

## Directory Structure

```
data/
├── facebook/
│   ├── facebook-1.90_dataset_description.txt  ✓ Included
│   ├── facebook-1.90.ungraph.txt              ✗ Download separately
│   ├── facebook-1.90.cmty.txt                ✗ Download separately
│   └── {k}-ego.txt                            Auto-generated (optional)
├── amazon/
│   ├── amazon-1.90_dataset_description.txt    ✓ Included
│   ├── amazon-1.90.ungraph.txt                ✗ Download separately
│   ├── amazon-1.90.cmty.txt                   ✗ Download separately
│   └── {k}-ego.txt                            Auto-generated (optional)
├── dblp/
│   ├── dblp-1.90_dataset_description.txt     ✓ Included
│   ├── dblp-1.90.ungraph.txt                  ✗ Download separately
│   ├── dblp-1.90.cmty.txt                    ✗ Download separately
│   └── {k}-ego.txt                            Auto-generated (optional)
├── lj/
│   ├── lj-1.90_dataset_description.txt       ✓ Included
│   ├── lj-1.90.ungraph.txt                   ✗ Download separately
│   ├── lj-1.90.cmty.txt                      ✗ Download separately
│   └── {k}-ego.txt                            Auto-generated (optional)
└── twitter/
    ├── twitter-1.90_dataset_description.txt  ✓ Included
    ├── twitter-1.90.ungraph.txt              ✗ Download separately
    ├── twitter-1.90.cmty.txt                ✗ Download separately
    └── {k}-ego.txt                            Auto-generated (optional)
```

## Getting the Data

### Option 1: Download from SNAP

The graph data files can be downloaded from the Stanford Network Analysis Project (SNAP):
- Website: https://snap.stanford.edu/data/
- Direct links for SNAP datasets:
  - Facebook: https://snap.stanford.edu/data/ego-Facebook.html
  - Amazon: https://snap.stanford.edu/data/com-Amazon.html
  - DBLP: https://snap.stanford.edu/data/com-DBLP.html
  - LiveJournal: https://snap.stanford.edu/data/com-LiveJournal.html
  - Twitter: https://snap.stanford.edu/data/ego-Twitter.html

### Option 2: Use Your Own Data

You can use your own graph data by:
1. Creating a directory under `data/` with your dataset name
2. Placing your edge list file as `{dataset_name}-1.90.ungraph.txt`
3. Placing your ground truth communities as `{dataset_name}-1.90.cmty.txt`
4. Creating a description file `{dataset_name}-1.90_dataset_description.txt` following the format of the included examples

## File Formats

### Edge List Format (`*.ungraph.txt`)
```
0 1
1 2
2 3
...
```
- One edge per line
- Space-separated node IDs
- Undirected graph (each edge appears once)

### Community Format (`*.cmty.txt`)
```
0 1 2 3
4 5 6
7 8 9 10 11
...
```
- One community per line
- Space-separated node IDs in that community
- Communities can overlap

### Dataset Description Format (`*_dataset_description.txt`)
See the included examples for the format. The description should include:
- Dataset overview
- Scale information (nodes, edges, communities)
- Community size statistics
- Any important notes about the dataset

## Important Notes

1. **Dataset Description Files**: These are **required** for the LLM-based approach to work effectively. They provide global context that helps the LLM generate better merge and refine rules.

2. **K-EGO Files**: The `{k}-ego.txt` files are optional. If they exist, the code will load them to save computation time. If they don't exist, the code will compute them automatically and save them for future use.

3. **File Naming**: The code expects specific file naming conventions:
   - Edge list: `{dataset}-1.90.ungraph.txt`
   - Communities: `{dataset}-1.90.cmty.txt`
   - Description: `{dataset}-1.90_dataset_description.txt`
   - K-ego: `{k}-ego.txt` (where k is the number of hops)

4. **Data Root**: By default, the code looks for data in `./data/`. You can specify a different location using the `--data_root` argument.
