import sys
import textwrap
import traceback
import argparse
import datetime
import math
import torch
import numpy as np
import time
import os
import random
import json
import re
import inspect
import multiprocessing as mp
from multiprocessing import Manager
from functools import partial
from joblib import Parallel, delayed
from torch_geometric.utils import k_hop_subgraph
from openai import OpenAI
from crowdkit.aggregation import DawidSkene
from crowdkit.datasets import load_dataset

import utils
from utils import *
from snap_datasets import *
from evaluate import *

API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4")


def fix_syntax_errors(code):
    """Fix common syntax errors in generated code"""
    # Fix common issues
    code = code.replace("len(B_edges')", "len(B_edges)")
    code = code.replace("len(A_edges')", "len(A_edges)")
    code = code.replace("len(B_nodes')", "len(B_nodes)")
    code = code.replace("len(A_nodes')", "len(A_nodes)")
    return code

# Global cache for rule functions and node mappings (per decision tree)
_rule_fn_cache = {}  # key: (tree_id, node_id) -> compiled function
_node_map_cache = {}  # key: tree_id -> dict mapping node_id -> node
_current_tree_id = [0]  # Use list to allow modification in nested scope

def evaluate_node_refine_tree(current_node, all_nodes, node_id, community_nodes, community_edges, node_labels, input_format="degree", precomputed_meta_base=None, node_map=None):
    """
    Recursively evaluate LLM decision tree to determine if a node should be kept in the community.
    
    Args:
        current_node: Current decision tree node
        all_nodes: All nodes in the decision tree
        node_id: Node ID being evaluated
        community_nodes: List of all nodes in the community
        community_edges: List of edges in the community
        node_labels: Dictionary mapping node_id -> label
        input_format: Input format ("degree" or "adjacency_list") to determine meta dictionary structure
        precomputed_meta_base: Pre-computed base meta dict (degrees or adj_list) to avoid recalculation
        node_map: Dict mapping node_id -> node for O(1) lookup
    
    Returns:
        bool: True if should keep, False if should remove
    """
    # If leaf node, return decision
    if current_node.get('kind') == 'leaf' or current_node.get('kind') == 'decision':
        node_id_str = current_node.get('node_id', '')
        return node_id_str == 'leaf_keep'
    
    # If rule node, evaluate rule
    if current_node.get('kind') == 'rule':
        rule_code = current_node.get('code', '')
        if not rule_code:
            return False
        
        # Execute the rule code
        try:
            # Use cached rule function if available
            rule_node_id = current_node.get('node_id', '')
            cache_key = (id(all_nodes), rule_node_id)  # Use all_nodes id as tree identifier
            
            if cache_key not in _rule_fn_cache:
                # Fix common syntax errors first
                rule_code_fixed = fix_syntax_errors(rule_code)
                
                # Create safe execution environment
                import math
                exec_globals = {
                    'len': len,
                    'set': set,
                    'abs': abs,
                    'min': min,
                    'max': max,
                    'sum': sum,
                    'sorted': sorted,
                    'range': range,
                    'bool': bool,
                    'int': int,
                    'float': float,
                    'math': math,
                }
                
                # Execute and cache the rule function
                exec(rule_code_fixed, exec_globals)
                rule_fn = exec_globals.get('rule_fn')
                _rule_fn_cache[cache_key] = rule_fn
            else:
                rule_fn = _rule_fn_cache[cache_key]
            
            if not rule_fn:
                return False
            
            # Create meta dictionary - reuse precomputed base if available
            if precomputed_meta_base is not None:
                meta = precomputed_meta_base.copy()
                meta['node_id'] = node_id
            else:
                # Fallback: compute on the fly (shouldn't happen with optimization)
                if input_format == "degree":
                    degrees = []
                    for node in community_nodes:
                        degree = len([n for n in community_nodes if (node, n) in community_edges or (n, node) in community_edges])
                        degrees.append(degree)
                    meta = {
                        'node_id': node_id,
                        'nodes': community_nodes,
                        'degrees': degrees,
                        'node_labels': node_labels,
                    }
                else:
                    adj_list = {}
                    for node in community_nodes:
                        adj_list[node] = [n for n in community_nodes if (node, n) in community_edges or (n, node) in community_edges]
                    meta = {
                        'node_id': node_id,
                        'nodes': community_nodes,
                        'adjacency_list': adj_list,
                        'node_labels': node_labels,
                    }
            
            result = rule_fn(meta)
            # Get next node based on result
            if result:
                next_node_id = current_node.get('true_child')
            else:
                next_node_id = current_node.get('false_child')
            
            # Find and evaluate next node - use node_map for O(1) lookup
            if node_map is not None:
                next_node = node_map.get(next_node_id)
            else:
                # Fallback: linear search
                next_node = None
                for node in all_nodes:
                    if node.get('node_id') == next_node_id:
                        next_node = node
                        break
            
            if next_node:
                return evaluate_node_refine_tree(next_node, all_nodes, node_id, community_nodes, community_edges, node_labels, input_format, precomputed_meta_base, node_map)
            else:
                # Leaf node reached
                return next_node_id == 'leaf_keep'
                
        except SyntaxError as e:
            print(f"Syntax error in node refine rule '{current_node.get('name')}': {e}", flush=True)
            print(f"Problematic code: {rule_code[:200]}...", flush=True)
            return False
        except Exception as e:
            print(f"Error evaluating node refine rule '{current_node.get('name')}': {e}", flush=True)
            return False
    
    return False

def evaluate_decision_tree(current_node, all_nodes, u, v, A_nodes, A_edges, B_nodes, B_edges, A_labels, B_labels, input_format="degree", precomputed_meta_base_A=None, precomputed_meta_base_B=None, node_map=None):
    """
    Recursively evaluate LLM decision tree to determine if two k-ego networks should merge.
    
    Args:
        current_node: Current decision tree node
        all_nodes: All nodes in the decision tree
        u, v: Node IDs
        A_nodes, B_nodes: Node lists of k-ego networks
        A_edges, B_edges: Edge lists of k-ego networks
        A_labels, B_labels: Lists of labels for each node in the subgraphs
        input_format: Input format ("degree" or "adjacency_list") to determine meta dictionary structure
        precomputed_meta_base_A: Pre-computed base meta dict for A (degrees or adj_list)
        precomputed_meta_base_B: Pre-computed base meta dict for B (degrees or adj_list)
        node_map: Dict mapping node_id -> node for O(1) lookup
    
    Returns:
        bool: True if should merge, False otherwise
    """
    # If leaf node, return decision
    if current_node.get('kind') == 'leaf':
        node_id = current_node.get('node_id', '')
        return node_id == 'leaf_merge'
    
    # If rule node, evaluate rule
    if current_node.get('kind') == 'rule':
        rule_code = current_node.get('code', '')
        if not rule_code:
            return False
        
        # Execute the rule code
        try:
            # Use cached rule function if available
            rule_node_id = current_node.get('node_id', '')
            cache_key = (id(all_nodes), rule_node_id)  # Use all_nodes id as tree identifier
            
            if cache_key not in _rule_fn_cache:
                # Fix common syntax errors first
                rule_code_fixed = fix_syntax_errors(rule_code)
                
                # Create safe execution environment
                import math
                exec_globals = {
                    'len': len,
                    'set': set,
                    'abs': abs,
                    'min': min,
                    'max': max,
                    'sum': sum,
                    'sorted': sorted,
                    'range': range,
                    'bool': bool,
                    'int': int,
                    'float': float,
                    'math': math,
                }
                
                # Execute and cache the rule function
                exec(rule_code_fixed, exec_globals)
                rule_fn = exec_globals.get('rule_fn')
                _rule_fn_cache[cache_key] = rule_fn
            else:
                rule_fn = _rule_fn_cache[cache_key]
            
            if not rule_fn:
                return False
            
            # Create meta dictionary - reuse precomputed base if available
            if precomputed_meta_base_A is not None and precomputed_meta_base_B is not None:
                meta = {
                    'A_nodes': A_nodes,
                    'B_nodes': B_nodes,
                    'A_labels': A_labels,
                    'B_labels': B_labels,
                    'u': u,
                    'v': v,
                }
                meta.update(precomputed_meta_base_A)
                meta.update(precomputed_meta_base_B)
            else:
                # Fallback: compute on the fly (shouldn't happen with optimization)
                if input_format == "degree":
                    # 使用度数而不是边列表
                    A_degrees = [len([n for n in A_nodes if (node, n) in A_edges or (n, node) in A_edges]) for node in A_nodes]
                    B_degrees = [len([n for n in B_nodes if (node, n) in B_edges or (n, node) in B_edges]) for node in B_nodes]
                    
                    meta = {
                        'A_nodes': A_nodes,
                        'B_nodes': B_nodes,
                        'A_degrees': A_degrees,
                        'B_degrees': B_degrees,
                        'A_labels': A_labels,
                        'B_labels': B_labels,
                        'u': u,
                        'v': v,
                    }
                else:
                    # 其他数据集: 使用邻接表而不是边列表
                    A_adj_list = {}
                    for node in A_nodes:
                        A_adj_list[node] = [n for n in A_nodes if (node, n) in A_edges or (n, node) in A_edges]
                    
                    B_adj_list = {}
                    for node in B_nodes:
                        B_adj_list[node] = [n for n in B_nodes if (node, n) in B_edges or (n, node) in B_edges]
                    
                    meta = {
                        'A_nodes': A_nodes,
                        'B_nodes': B_nodes,
                        'A_adjacency_list': A_adj_list,
                        'B_adjacency_list': B_adj_list,
                        'A_labels': A_labels,
                        'B_labels': B_labels,
                        'u': u,
                        'v': v,
                    }
            
            result = rule_fn(meta)
            # Get next node based on result
            if result:
                next_node_id = current_node.get('true_child')
            else:
                next_node_id = current_node.get('false_child')
            
            # Find and evaluate next node - use node_map for O(1) lookup
            if node_map is not None:
                next_node = node_map.get(next_node_id)
            else:
                # Fallback: linear search
                next_node = None
                for node in all_nodes:
                    if node.get('node_id') == next_node_id:
                        next_node = node
                        break
            
            if next_node:
                return evaluate_decision_tree(next_node, all_nodes, u, v, A_nodes, A_edges, B_nodes, B_edges, A_labels, B_labels, input_format, precomputed_meta_base_A, precomputed_meta_base_B, node_map)
            else:
                # Leaf node reached
                return next_node_id == 'leaf_merge'
                
        except SyntaxError as e:
            print(f"Syntax error in rule '{current_node.get('name')}': {e}", flush=True)
            print(f"Problematic code: {rule_code[:200]}...", flush=True)
            return False
        except Exception as e:
            print(f"Error evaluating rule '{current_node.get('name')}': {e}", flush=True)
            return False
    
    return False


def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets"""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def merge_prompt_v2(dataset, num_node, num_edge, communities, k_ego_samples, input_format="degree", tree_depth_range=(8, 12), data_root="./data"):
    """
    Generate merge prompt based on k-ego samples.
    
    Args:
        dataset: Dataset name
        num_node: Number of nodes
        num_edge: Number of edges
        communities: List of communities
        k_ego_samples: List of k-ego samples
        input_format: Input format ("degree" or "adjacency_list")
        tree_depth_range: Range of tree depth
        data_root: Root directory for data files
    """
    description_path = os.path.join(data_root, dataset, f"{dataset}-1.90_dataset_description.txt")
    dataset_description = ""
    if os.path.exists(description_path):
        with open(description_path, 'r', encoding='utf-8') as f:
            dataset_description = f.read().strip()
    else:
        print(f"Warning: Dataset description file not found: {description_path}")
    
    system_content = """\
You are an expert in graph community detection and structure reasoning.
Your task is to analyze k-ego network samples and generate:
1. A JSON decision tree for merge decisions，the resulting rules must be precise, conservative, and multi-dimensional
2. A Python function for executing merge operations
    """

    # ---- 生成 K-EGO 样例信息块 ----
    sample_blocks = []
    for i, (center_node, center_k_ego_nodes, center_k_ego_edges, center_k_ego_labels, neighbors_list) in enumerate(k_ego_samples):
        # neighbors_list: [(neighbor_node, neighbor_k_ego_nodes, neighbor_k_ego_edges, neighbor_k_ego_labels), ...]
        neighbors_info = []
        for j, (neighbor_node, neighbor_k_ego_nodes, neighbor_k_ego_edges, neighbor_k_ego_labels) in enumerate(neighbors_list):
            if input_format == "degree":
                neighbor_degrees = [len([n for n in neighbor_k_ego_nodes if (node, n) in neighbor_k_ego_edges or (n, node) in neighbor_k_ego_edges]) for node in neighbor_k_ego_nodes]
                neighbors_info.append(f"""
  Neighbor {j} (node {neighbor_node}):
    k_ego_nodes = {neighbor_k_ego_nodes}
    k_ego_degrees = {neighbor_degrees}
    k_ego_labels = {neighbor_k_ego_labels}
""")
            else:  # adjacency_list
                # 构建邻接表
                neighbor_adj_list = {}
                for node in neighbor_k_ego_nodes:
                    neighbor_adj_list[node] = [n for n in neighbor_k_ego_nodes if (node, n) in neighbor_k_ego_edges or (n, node) in neighbor_k_ego_edges]
                neighbors_info.append(f"""
  Neighbor {j} (node {neighbor_node}):
    k_ego_nodes = {neighbor_k_ego_nodes}
    k_ego_adjacency_list = {neighbor_adj_list}
    k_ego_labels = {neighbor_k_ego_labels}
""")
        
        neighbors_block = "".join(neighbors_info)
        
        if input_format == "degree":
            center_degrees = [len([n for n in center_k_ego_nodes if (node, n) in center_k_ego_edges or (n, node) in center_k_ego_edges]) for node in center_k_ego_nodes]
            block = f"""
[SAMPLE {i}]
Center Node: {center_node}
  k_ego_nodes = {center_k_ego_nodes}
  k_ego_degrees = {center_degrees}
  k_ego_labels = {center_k_ego_labels}
  k_ego_size = {len(center_k_ego_nodes)}
  direct_neighbors_count = {len(neighbors_list)}
{neighbors_block}
"""
        else:  # adjacency_list
            # 构建中心节点的邻接表
            center_adj_list = {}
            for node in center_k_ego_nodes:
                center_adj_list[node] = [n for n in center_k_ego_nodes if (node, n) in center_k_ego_edges or (n, node) in center_k_ego_edges]
            block = f"""
[SAMPLE {i}]
Center Node: {center_node}
  k_ego_nodes = {center_k_ego_nodes}
  k_ego_adjacency_list = {center_adj_list}
  k_ego_labels = {center_k_ego_labels}
  k_ego_size = {len(center_k_ego_nodes)}
  direct_neighbors_count = {len(neighbors_list)}
{neighbors_block}
"""
        sample_blocks.append(block.strip())

    samples_block = "\n\n".join(sample_blocks)

    # Extract tree depth range
    min_tree_nodes, max_tree_nodes = tree_depth_range

    # ---- 用户 Prompt ----
    user_content = f"""\
[DATASET INFO]
{dataset_description}

[K-EGO SAMPLES]
Below are {len(k_ego_samples)} sampled nodes with their k-ego networks and direct neighbors' k-ego networks:
{samples_block}

[TASK INSTRUCTION]  
You are to analyze all k-ego samples jointly and:
1.Induce a multi-factor decision tree ({min_tree_nodes}–{max_tree_nodes} rule nodes) that determines, for the given center node, which of its neighboring k-ego networks should belong to the same community and therefore should be merged with the center node's k-ego network.
2.generate a merge execution function that merges the subgraphs into one community based on the decision tree.
**Note:** The merge execution function should directly implement the decision tree logic with optimized computation. Since traversing the decision tree would require recalculating metrics multiple times, the function should pre-compute shared metrics once and reuse them across multiple neighbor evaluations to minimize time complexity.

---
### Step 0: Dataset-level Awareness
Before analyzing samples, use dataset_description to maintain a reasonable overall community scale — avoid merging so aggressively that the total community count drops drastically, and avoid being so conservative that almost no merges occur.
Adjust merge thresholds accordingly to achieve balanced community refinement.


### Step 1: Generate multi-factor decision tree for merge decisions

## Step 1.1: Metric Definition and Calculation
For each center node and its multiple neighbor k-ego networks, **define and compute** interpretable metrics that capture structural and semantic relations between the center's k-ego network and each neighbor's k-ego network.  
**Important:** This is NOT a pairwise comparison task. Instead, you are evaluating multiple neighbor candidates (one center + multiple neighbors) to determine which neighbors should merge with the center. Each decision is made independently based on the pair consisting of a center node's k-ego network and the k-ego network of one of its direct neighbors.

Do **not** assume fixed formulas — instead, write your own **Python-style code** to compute each metric explicitly inside the decision tree nodes.

Each metric should reflect an intuitive concept such as:
- Node **overlap** between the center's k-ego network and a neighbor's k-ego network
- Internal **density** or **compactness** of each k-ego network (center's and neighbor's)
- **Density similarity** between the center's and neighbor's k-ego networks
- **Size balance** between the center's and neighbor's k-ego networks
- **Semantic agreement** between nodes in the center's and neighbor's networks
- **Inter-edge connectivity** between the center's and neighbor's k-ego networks
- **Merged cohesion** (connectivity and compactness after union of center and neighbor)

Your code should show **how each metric is calculated** from:
- When input_format is "degree": `center_k_ego_nodes`, `neighbor_k_ego_nodes`, `center_k_ego_degrees`, `neighbor_k_ego_degrees`, `center_k_ego_labels`, `neighbor_k_ego_labels` (where degrees are lists of node degrees and labels are lists of labels for each node)
- When input_format is "adjacency_list": `center_k_ego_nodes`, `neighbor_k_ego_nodes`, `center_k_ego_adjacency_list`, `neighbor_k_ego_adjacency_list`, `center_k_ego_labels`, `neighbor_k_ego_labels` (where adjacency_list is a dict mapping node_id to list of neighbors and labels are lists of labels for each node)

You may create intermediate variables and helper computations if needed.  
The goal is to make every metric **transparent and interpretable** rather than relying on fixed formulas.

After defining these metrics, analyze all center-neighbor k-ego network combinations to identify:
- **Merge-worthy patterns** (high overlap, consistent labels, balanced structures)
- **Non-merge patterns** (low connectivity, structural dissimilarity, semantic mismatch)

**Note:** For each center node, you will evaluate multiple neighbors independently. The decision tree should be designed to handle this one-center-multiple-neighbors scenario.

---

## Step 1.2: Conservative Rule Induction  
Generate a **structured decision tree** with {min_tree_nodes}–{max_tree_nodes} rule nodes.  
Each node should evaluate one interpretable metric and decide whether to continue to a deeper condition or stop.
**Each rule node must include:**
- `"node_id"`: unique identifier  
- `"kind": "rule"`  
- `"name"`: concise metric name  
- `"code"`: Python-like function implementing the metric calculation and returning `True` if merge is favored  
- `"true_child"` / `"false_child"`: next rule or leaf node

Example node format (when input_format is "degree"):
```python
{{
  "node_id": "root",
  "kind": "rule",
  "name": "check_overlap_density",
  "code": "def rule_fn(meta):
    center_nodes = meta['center_k_ego_nodes']
    neighbor_nodes = meta['neighbor_k_ego_nodes']
    center_labels = meta['center_k_ego_labels']
    neighbor_labels = meta['neighbor_k_ego_labels']
    center_degrees = meta['center_k_ego_degrees']
    neighbor_degrees = meta['neighbor_k_ego_degrees']
    overlap = len(set(center_nodes) & set(neighbor_nodes))
    union = len(set(center_nodes) | set(neighbor_nodes))
    node_overlap_ratio = overlap / union if union > 0 else 0
    # Check label agreement
    label_agreement = sum(1 for i, label in enumerate(center_labels) if i < len(neighbor_labels) and label == neighbor_labels[i]) / max(len(center_labels), len(neighbor_labels)) if len(center_labels) > 0 else 0
    # Check density similarity using degrees
    avg_degree_center = sum(center_degrees) / len(center_degrees) if len(center_degrees) > 0 else 0
    avg_degree_neighbor = sum(neighbor_degrees) / len(neighbor_degrees) if len(neighbor_degrees) > 0 else 0
    density_similarity = 1.0 - abs(avg_degree_center - avg_degree_neighbor) / max(avg_degree_center, avg_degree_neighbor, 1)
    return node_overlap_ratio > 0.5 and label_agreement > 0.7 and density_similarity > 0.6",
  "true_child": "check_label_consistency",
  "false_child": "leaf_no_merge"
}}
```

Example node format (when input_format is "adjacency_list"):
```python
{{
  "node_id": "root",
  "kind": "rule",
  "name": "check_overlap_density",
  "code": "def rule_fn(meta):
    center_nodes = meta['center_k_ego_nodes']
    neighbor_nodes = meta['neighbor_k_ego_nodes']
    center_labels = meta['center_k_ego_labels']
    neighbor_labels = meta['neighbor_k_ego_labels']
    center_adj = meta['center_k_ego_adjacency_list']
    neighbor_adj = meta['neighbor_k_ego_adjacency_list']
    overlap = len(set(center_nodes) & set(neighbor_nodes))
    union = len(set(center_nodes) | set(neighbor_nodes))
    node_overlap_ratio = overlap / union if union > 0 else 0
    # Check label agreement
    label_agreement = sum(1 for i, label in enumerate(center_labels) if i < len(neighbor_labels) and label == neighbor_labels[i]) / max(len(center_labels), len(neighbor_labels)) if len(center_labels) > 0 else 0
    # Check density using adjacency lists
    center_edges = sum(len(neighbors) for neighbors in center_adj.values()) // 2
    neighbor_edges = sum(len(neighbors) for neighbors in neighbor_adj.values()) // 2
    center_density = center_edges / (len(center_nodes) * (len(center_nodes) - 1) / 2) if len(center_nodes) > 1 else 0
    neighbor_density = neighbor_edges / (len(neighbor_nodes) * (len(neighbor_nodes) - 1) / 2) if len(neighbor_nodes) > 1 else 0
    density_similarity = 1.0 - abs(center_density - neighbor_density) / max(center_density, neighbor_density, 0.01)
    return node_overlap_ratio > 0.5 and label_agreement > 0.7 and density_similarity > 0.6",
  "true_child": "check_label_consistency",
  "false_child": "leaf_no_merge"
}}
```

**Note:** The `meta` dictionary in rule functions will contain all necessary data based on input_format:
- When input_format is "degree": `center_k_ego_nodes`, `neighbor_k_ego_nodes`, `center_k_ego_degrees`, `neighbor_k_ego_degrees`, `center_k_ego_labels`, `neighbor_k_ego_labels`
- When input_format is "adjacency_list": `center_k_ego_nodes`, `neighbor_k_ego_nodes`, `center_k_ego_adjacency_list`, `neighbor_k_ego_adjacency_list`, `center_k_ego_labels`, `neighbor_k_ego_labels`

Design the tree with a **higher merge threshold** — merging should only occur when multiple metrics jointly support it.  
Favor **precision over recall** in merge decisions.

**Leaf node rules:**
- `"leaf_merge"` → merging is strongly supported. This leaf node indicates that the neighbor k-ego network should be merged with the center k-ego network. The merge result should include all nodes from both networks: `list(set(center_k_ego_nodes) | set(neighbor_k_ego_nodes))`
- `"leaf_no_merge"` → merging is not recommended. This leaf node indicates that the neighbor k-ego network should NOT be merged, and only the center k-ego network nodes should be kept.

**Important:** Unlike pairwise subgraph comparison, this task involves multiple k-ego networks (one center + multiple neighbors). Each decision evaluates whether a specific neighbor's k-ego network should merge with the center's. 

---

## Step 1.3: Decision Path Simulation  
After constructing the decision tree based on all provided samples, **demonstrate the decision process** by simulating a few representative examples.  
The decision tree you generate should be a **general rule** that can be applied to any center-neighbor pair, learned from analyzing all the provided samples collectively.  
For demonstration purposes, select a few representative samples and trace the decision paths for some of their neighbors to show how the tree works.

For each decision in `sample_decisions`, include:
- `"sample_id"`: the sample index (which center node, 0-indexed from the provided samples)
- `"neighbor_id"`: the ID of the neighbor node being evaluated
- `"merge"`: boolean indicating whether to merge (True if path ends at `leaf_merge`, False if ends at `leaf_no_merge`)
- `"path"`: list of node_ids traversed from root to leaf
- `"merged_nodes"`: if merge=True, this should be the list of all node IDs after merging (union of center_k_ego_nodes and neighbor_k_ego_nodes); if merge=False, this should be just the center_k_ego_nodes
- `"reason"`: concise summary of the decisive factors

**Important:** 
- The decision tree should be learned from **analyzing ALL {len(k_ego_samples)} provided samples** to understand patterns across different center nodes and their neighbors.
- `sample_decisions` is for **demonstration purposes only** - include a few representative examples (e.g., 5-10 decisions) showing different merge/no-merge scenarios.
- The `merge_execution_function` you generate will be applied to **ALL neighbors** of each center node during actual execution, not just the examples shown in `sample_decisions`.

Important:

Produce a decision tree whose rule nodes each contain **self-contained Python code** that defines and evaluates a meaningful structural metric, allowing for reproducible, explainable merge decisions.

Keep rules focused and modular.

### Step 2: Generate Merge Execution Function
Convert the decision tree into a direct Python function that processes multiple neighbor k-ego networks.

**Function signature (must match input_format):**

When input_format is "degree":
def merge_k_ego_networks(center_k_ego_nodes, center_k_ego_degrees, center_k_ego_labels, neighbor_k_ego_networks, current_community=None):
    # Input:
    #   center_k_ego_nodes: list[int] - nodes in the center node's k-ego network
    #   center_k_ego_degrees: list[int] - degrees of nodes in center_k_ego_nodes (same order)
    #   center_k_ego_labels: list - labels of nodes in center_k_ego_nodes (same order)
    #   neighbor_k_ego_networks: list[tuple] - list of (neighbor_node_id, neighbor_k_ego_nodes, neighbor_k_ego_degrees, neighbor_k_ego_labels) tuples
    #   current_community: list[int] - current community nodes (if merging incrementally)
    # Output:

When input_format is "adjacency_list":
def merge_k_ego_networks(center_k_ego_nodes, center_k_ego_adjacency_list, center_k_ego_labels, neighbor_k_ego_networks, current_community=None):
    # Input:
    #   center_k_ego_nodes: list[int] - nodes in the center node's k-ego network
    #   center_k_ego_adjacency_list: dict[int, list[int]] - adjacency list for center k-ego network (node_id -> list of neighbors)
    #   center_k_ego_labels: list - labels of nodes in center_k_ego_nodes (same order)
    #   neighbor_k_ego_networks: list[tuple] - list of (neighbor_node_id, neighbor_k_ego_nodes, neighbor_k_ego_adjacency_list, neighbor_k_ego_labels) tuples
    #   current_community: list[int] - current community nodes (if merging incrementally)
    # Output:

Requirements:
- Direct implementation (no tree traversal overhead)
- The function must receive all necessary data (nodes, labels, degrees/adjacency_list) to compute the metrics defined in the decision tree
- Efficient set operations for overlap calculation: overlap = len(set(center_k_ego_nodes) & set(neighbor_k_ego_nodes))
- Process each neighbor k-ego network independently and decide whether to merge based on the decision tree logic
- Return merged node list: union of center_k_ego_nodes and all selected neighbor_k_ego_nodes
- If current_community is provided, start from it instead of just center_k_ego_nodes
- No print statements, no exception handling
- **Code length limit: The function code must not exceed 100 lines**

### Step 3: Structured Output
Return a **single valid JSON** object:
```json
{{
  "merge_decision_tree": {{
    "nodes": [
      {{
        "node_id": "root",
        "kind": "rule",
        "name": "check_overlap",
        "code": "def rule_fn(meta):\\n    center_nodes = meta['center_k_ego_nodes']\\n    neighbor_nodes = meta['neighbor_k_ego_nodes']\\n    overlap = len(set(center_nodes) & set(neighbor_nodes))\\n    union = len(set(center_nodes) | set(neighbor_nodes))\\n    overlap_ratio = overlap / union if union > 0 else 0\\n    return overlap_ratio > 0.3 and overlap >= 3",
        "true_child": "check_size",
        "false_child": "leaf_no_merge"
      }},
      {{
        "node_id": "check_size",
        "kind": "rule",
        "name": "check_community_size",
        "code": "def rule_fn(meta):\\n    center_nodes = meta['center_k_ego_nodes']\\n    neighbor_nodes = meta['neighbor_k_ego_nodes']\\n    current = meta.get('current_community', center_nodes)\\n    merged_size = len(set(current) | set(neighbor_nodes))\\n    return merged_size < 30",
        "true_child": "leaf_merge",
        "false_child": "leaf_no_merge"
      }},
      {{
        "node_id": "leaf_merge",
        "kind": "leaf",
        "name": "merge_neighbor",
        "action": "merge"
      }},
      {{
        "node_id": "leaf_no_merge",
        "kind": "leaf",
        "name": "keep_separate",
        "action": "no_merge"
      }}
    ],
    "sample_decisions": [
      {{
        "sample_id": 0,
        "neighbor_id": 15,
        "merge": true,
        "path": ["root", "check_size", "leaf_merge"],
        "merged_nodes": [1, 2, 3, 4, 5, 15, 16, 17],
        "reason": "High overlap ratio (0.45) and merged size (18) within threshold"
      }},
      {{
        "sample_id": 0,
        "neighbor_id": 23,
        "merge": false,
        "path": ["root", "leaf_no_merge"],
        "merged_nodes": [1, 2, 3, 4, 5],
        "reason": "Low overlap ratio (0.12), insufficient connectivity"
      }}
    ]
  }},
  "merge_execution_function": {{
    "code_degree": "def merge_k_ego_networks(center_k_ego_nodes, center_k_ego_degrees, center_k_ego_labels, neighbor_k_ego_networks, current_community=None):\\n    merged = set(current_community) if current_community else set(center_k_ego_nodes)\\n    for neighbor_id, neighbor_nodes, neighbor_degrees, neighbor_labels in neighbor_k_ego_networks:\\n        neighbor_set = set(neighbor_nodes)\\n        overlap = len(merged & neighbor_set)\\n        union = len(merged | neighbor_set)\\n        overlap_ratio = overlap / union if union > 0 else 0\\n        if overlap_ratio > 0.3 and overlap >= 3 and len(merged | neighbor_set) < 30:\\n            merged |= neighbor_set\\n    return list(merged)",
    "code_adjacency_list": "def merge_k_ego_networks(center_k_ego_nodes, center_k_ego_adjacency_list, center_k_ego_labels, neighbor_k_ego_networks, current_community=None):\\n    merged = set(current_community) if current_community else set(center_k_ego_nodes)\\n    for neighbor_id, neighbor_nodes, neighbor_adj_list, neighbor_labels in neighbor_k_ego_networks:\\n        neighbor_set = set(neighbor_nodes)\\n        overlap = len(merged & neighbor_set)\\n        union = len(merged | neighbor_set)\\n        overlap_ratio = overlap / union if union > 0 else 0\\n        if overlap_ratio > 0.3 and overlap >= 3 and len(merged | neighbor_set) < 30:\\n            merged |= neighbor_set\\n    return list(merged)"
  }}
}}
```

**Code Style Requirements:**
- **Concise**: No unnecessary variables or operations
- **Efficient**: Use set operations (O(n)) instead of list operations (O(n²))
- **Direct**: No intermediate data structures unless necessary
- **Clean**: No comments, no print statements, no try-except blocks
- **Fast**: Minimize time complexity (prefer O(n) over O(n log n) when possible)

**Important:**
- Merge execution function should be a direct, optimized version of the decision tree logic
- **The merge execution function MUST receive all necessary inputs** (nodes, labels, degrees/adjacency_list) to compute the metrics used in the decision tree
- The function must be directly executable Python code (no pseudocode)
- **Code length limit: The function (merge_k_ego_networks) must not exceed 100 lines**
- **Remember:** This is a one-center-multiple-neighbors scenario, not pairwise comparison. The function processes multiple neighbors for a single center node.

Return ONLY the JSON (no explanations).
"""
    messages = [
        {"role": "system", "content": system_content.strip()},
        {"role": "user", "content": user_content.strip()},
    ]

    # debug print (nice readable): can remove once everything works
    print("=== 当前生成的 messages (merge_prompt_v2) ===")
    print(json.dumps(messages, indent=2, ensure_ascii=False))
    print("=== 以上是 merge_prompt_v2 的输出 ===")

    return messages

def merge_prompt(dataset, num_node, num_edge, communities, subgraph_pairs, input_format="degree", data_root="./data"):
    """
    Generate merge prompt (legacy version, kept for compatibility).
    
    Args:
        dataset: Dataset name
        num_node: Number of nodes
        num_edge: Number of edges
        communities: List of communities
        subgraph_pairs: List of subgraph pairs
        input_format: Input format ("degree" or "adjacency_list")
        data_root: Root directory for data files
    """
    # Read dataset description file
    description_path = os.path.join(data_root, dataset, f"{dataset}-1.90_dataset_description.txt")
    dataset_description = ""
    if os.path.exists(description_path):
        with open(description_path, 'r', encoding='utf-8') as f:
            dataset_description = f.read().strip()
    else:
        print(f"Warning: Dataset description file not found: {description_path}")
    
    system_content = """\
You are an expert in graph community detection and structure reasoning.
Your task is to derive interpretable and generalizable merge rules that decide
whether two candidate subgraphs should be merged into one community.
The resulting rules must be precise, conservative, and multi-dimensional
    """

    # ---- 生成 Subgraph Pair 信息块 ----
    pair_blocks = []
    for i, (u, v, A_nodes, A_edges, B_nodes, B_edges, A_labels, B_labels) in enumerate(subgraph_pairs):
        if input_format == "degree":
            # 使用node list, node degree, label list
            A_degrees = [len([n for n in A_nodes if (node, n) in A_edges or (n, node) in A_edges]) for node in A_nodes]
            B_degrees = [len([n for n in B_nodes if (node, n) in B_edges or (n, node) in B_edges]) for node in B_nodes]
            
            block = f"""
[PAIR {i}]
Subgraph Pair ({u}, {v}):
Subgraph A:
  nodes = {A_nodes}
  degrees = {A_degrees}
  labels = {A_labels}
Subgraph B:
  nodes = {B_nodes}
  degrees = {B_degrees}
  labels = {B_labels}
"""
        else:
            # 其他数据集: 使用node list, 邻接表, label list
            A_adj_list = {}
            for node in A_nodes:
                A_adj_list[node] = [n for n in A_nodes if (node, n) in A_edges or (n, node) in A_edges]
            
            B_adj_list = {}
            for node in B_nodes:
                B_adj_list[node] = [n for n in B_nodes if (node, n) in B_edges or (n, node) in B_edges]
            
            block = f"""
[PAIR {i}]
Subgraph Pair ({u}, {v}):
Subgraph A:
  nodes = {A_nodes}
  adjacency_list = {A_adj_list}
  labels = {A_labels}
Subgraph B:
  nodes = {B_nodes}
  adjacency_list = {B_adj_list}
  labels = {B_labels}
"""
        pair_blocks.append(block.strip())

    subgraph_block = "\n\n".join(pair_blocks)

    # ---- 用户 Prompt ----
    #改进版本v2,由于v1没有用，现明示llm生成差不多数量和大小的候选社区，看看效果,v3版本进一步引导llm关于数据集全局信息的感知,v4增添了关于nodelabel explain(已删除)
    user_content = f"""\
[DATASET INFO]
{dataset_description}

[SUBGRAPH PAIRS INFO]
Below are {len(subgraph_pairs)} subgraph pairs sampled from the dataset:
{subgraph_block}

[TASK INSTRUCTION]  
You are to analyze all subgraph pairs jointly and induce a **multi-factor decision tree (8–12 rule nodes)**  
that determines whether two subgraphs should be merged into one community.

---
### Step 0: Dataset-level Awareness
Before analyzing subgraph pairs, use this context to maintain a reasonable overall community scale — avoid merging so aggressively that the total community count drops drastically, and avoid being so conservative that almost no merges occur.
Adjust merge thresholds accordingly to achieve balanced community refinement.


### Step 1: Metric Definition and Calculation
For each subgraph pair, **define and compute** interpretable metrics that capture structural and semantic relations between the two subgraphs.  
Do **not** assume fixed formulas — instead, write your own **Python-style code** to compute each metric explicitly inside the decision tree nodes.

Each metric should reflect an intuitive concept such as:
- Node or edge **overlap**
- Internal **density** or **compactness**
- **Density similarity** between subgraphs
- **Size balance** between A and B
- **Semantic or label agreement**
- **Inter-edge connectivity** between A and B
- **Merged cohesion** (connectivity and compactness after union)

Your code should show **how each metric is calculated** from:
- When input_format is "degree": `A_nodes`, `B_nodes`, `A_degrees`, `B_degrees`, `A_labels`, `B_labels` (where degrees are lists of node degrees and labels are lists of labels for each node)
- When input_format is "adjacency_list": `A_nodes`, `B_nodes`, `A_adjacency_list`, `B_adjacency_list`, `A_labels`, `B_labels` (where adjacency_list is a dict mapping node_id to list of neighbors and labels are lists of labels for each node)

You may create intermediate variables and helper computations if needed.  
The goal is to make every metric **transparent and interpretable** rather than relying on fixed formulas.

After defining these metrics, analyze all subgraph pairs to identify:
- **Merge-worthy patterns** (high overlap, consistent labels, balanced structures)
- **Non-merge patterns** (low connectivity, structural dissimilarity, semantic mismatch)

---

### Step 2: Conservative Rule Induction  
Generate a **structured decision tree** with 8–12 rule nodes.  
Each node should evaluate one interpretable metric and decide whether to continue to a deeper condition or stop.

**Each rule node must include:**
- `"node_id"`: unique identifier  
- `"kind": "rule"`  
- `"name"`: concise metric name  
- `"code"`: Python-like function implementing the metric calculation and returning `True` if merge is favored  
- `"true_child"` / `"false_child"`: next rule or leaf node

Example node format:
```python
{{
  "node_id": "root",
  "kind": "rule",
  "name": "check_overlap_density",
      "code": "def rule_fn(meta):
    A_nodes, B_nodes = meta['A_nodes'], meta['B_nodes']
    A_labels, B_labels = meta['A_labels'], meta['B_labels']
    overlap = len(set(A_nodes) & set(B_nodes))
    union = len(set(A_nodes) | set(B_nodes))
    node_overlap_ratio = overlap / union if union > 0 else 0
    # Check label agreement
    label_agreement = sum(1 for i, label in enumerate(A_labels) if i < len(B_labels) and label == B_labels[i]) / max(len(A_labels), len(B_labels))
    return node_overlap_ratio > 0.5 and label_agreement > 0.7",
  "true_child": "check_label_consistency",
  "false_child": "leaf_no_merge"
}}
```

Design the tree with a **higher merge threshold** — merging should only occur when multiple metrics jointly support it.  
Favor **precision over recall** in merge decisions.

Leaf node rules:
- `"leaf_merge"` → merging is strongly supported  
- `"leaf_no_merge"` → merging is not recommended  

---

### Step 3: Decision Path Simulation  
After constructing the decision tree, **simulate the decision process** for each provided subgraph pair.  
For each sample in `"sample_decisions"`, trace the **exact path** from root to leaf that the pair follows in the tree.  
The `"path"` list must contain valid `"node_id"` strings defined in `"nodes"`, following logical execution order.

Each decision must also include a concise `"reason"` summarizing the decisive factors.

---

### Step 4: Structured Output  
Return a **single valid JSON** object in the following format:
```json
{{
  "nodes": [
    {{
      "node_id": "root",
      "kind": "rule",
      "name": "check_overlap_density",
      "code": "def rule_fn(meta):
        A_nodes, B_nodes = meta['A_nodes'], meta['B_nodes']
        A_labels, B_labels = meta['A_labels'], meta['B_labels']
        overlap = len(set(A_nodes) & set(B_nodes))
        union = len(set(A_nodes) | set(B_nodes))
        node_overlap_ratio = overlap / union if union > 0 else 0
        density_A = len(meta['A_edges']) / (len(A_nodes) * (len(A_nodes) - 1) / 2) if len(A_nodes) > 1 else 0
        density_B = len(meta['B_edges']) / (len(B_nodes) * (len(B_nodes) - 1) / 2) if len(B_nodes) > 1 else 0
        label_agreement = sum(1 for i, label in enumerate(A_labels) if i < len(B_labels) and label == B_labels[i]) / max(len(A_labels), len(B_labels)) if len(A_labels) > 0 else 0
        return node_overlap_ratio > 0.45 and abs(density_A - density_B) < 0.1 and label_agreement > 0.7",
      "true_child": "check_label_consistency",
      "false_child": "leaf_no_merge"
    }}
  ],
  "sample_decisions": [
    {{"pair_id": 3, "merge": true, "path": ["root", "check_label_consistency", "leaf_merge"]}},
    {{"pair_id": 9, "merge": false, "path": ["root", "check_density_similarity", "leaf_no_merge"]}}
  ]
}}
```
Important:

Produce a decision tree whose rule nodes each contain **self-contained Python code** that defines and evaluates a meaningful structural metric, allowing for reproducible, explainable merge decisions.

Keep rules focused and modular.

Return ONLY the JSON (no explanations).
"""
    messages = [
        {"role": "system", "content": system_content.strip()},
        {"role": "user", "content": user_content.strip()},
    ]

    # debug print (nice readable): can remove once everything works
    print("=== 当前生成的 messages ===")
    print(json.dumps(messages, indent=2, ensure_ascii=False))
    print("=== 以上是 merge_prompt 的输出 ===")

    return messages

def refine_prompt(candidate_communities, dataset_name, input_format="adjacency_list", data_root="./data"):
    """
    Generate refine prompt for node refinement.
    
    Args:
        candidate_communities: List of candidate communities
        dataset_name: Dataset name
        input_format: Input format ("degree" or "adjacency_list")
        data_root: Root directory for data files
    """
    description_path = os.path.join(data_root, dataset_name, f"{dataset_name}-1.90_dataset_description.txt")
    
    # Read dataset description file
    dataset_description = ""
    if os.path.exists(description_path):
        with open(description_path, 'r', encoding='utf-8') as f:
            dataset_description = f.read().strip()
    else:
        print(f"Warning: Dataset description file not found: {description_path}")

    system_content = f"""
You are an expert in graph community detection. Your task is to generate a
**single Python function (< 100 lines)** for large-scale node-level boundary
refinement on **tens of thousands (10k+) of communities**.
These communities may come from:
The k-ego networks or merged k-ego networks contain most of the true community information but also include some noisy nodes, so it is necessary to remove nodes that do not belong to the community.
"""
    user_content = f"""
### GLOBAL DATASET INFORMATION SOURCE (IMPORTANT)
{dataset_description}

### INPUT REQUIREMENTS
You will receive **{len(candidate_communities)} sampled candidate communities**.
They are only EXAMPLES for guiding your refinement strategy.
At real runtime, your generated code will process 10k+ communities.

### HARD REQUIREMENTS
1. **CPU-only, NO GPU**  
   - Use ONLY pure Python + NumPy  
   - DO NOT import torch / cuda / cupy / numba / multiprocessing  
2. **Time complexity must remain O(n) – O(n log n)**.  
3. Function must handle:
   - adjacency_list OR degree input
   - optional node_labels
   - missing keys, empty communities
4. No shape/broadcast errors:  
   - Validate array lengths before operations  
   - Avoid mismatched advanced indexing  
5. Output format MUST be:  
   `dict[int, list[int]]` (community index → nodes_to_drop)

### LOGIC REQUIREMENTS
Your refinement decision should combine:
- **Structural signals**: degree, low-core nodes, leaves  
- **Topological signals**: weak triadic support  
- **Label signals** (if available): neighbor agreement, majority label alignment  


### YOUR TASK
Write **one** Python function:

```python
def batch_refine_communities(communities):
    '''
    communities: list of dicts, each containing:
        - 'nodes': list[int]
        - 'adjacency_list' OR 'degree'
        - 'node_labels' (optional): dict mapping node_id to label (int -> int)
    Returns:
        dict[int, list[int]]: community index -> nodes_to_drop
    '''

### CODE STYLE RULES
- < 100 lines total  
- No classes, no helper modules (helpers defined inline OK)  
- Compact but readable  
- Robust against invalid inputs  
- No unnecessary printing or comments  

The output must be actual runnable Python code, not pseudocode.
Generate only the function body.
"""
    messages = [
        {"role": "system", "content": system_content.strip()},
        {"role": "user", "content": user_content.strip()},
    ]
    print("=== 当前生成的 messages ===")
    print(json.dumps(messages, indent=2, ensure_ascii=False))
    print("=== 以上是 merge_prompt 的输出 ===")

    return messages

def evaluate(pred_comms, test_comms):
    scores_1, scores_2 = eval_comms_bidirectional(pred_comms, test_comms)
    mean_score_1 = scores_1.mean(0)
    print_results(mean_score_1, 'AvgOverAxis0')
    mean_score_2 = scores_2.mean(0)
    print_results(mean_score_2, 'AvgOverAxis1')
    mean_score_all = (mean_score_1 + mean_score_2) / 2.
    print_results(mean_score_all, 'AvgGlobal')
    return mean_score_1, mean_score_2, mean_score_all

def get_intersection(a, b, choice=None):
    intersect = list(set(a) & set(b))
    return intersect if choice == "List" else len(intersect)

def get_difference(a, b):
    intersection = get_intersection(a, b, choice="List")
    nodes = {x for x in a if x not in intersection}
    return len(list(nodes))

def get_nmi_score(pred, gt):
    def get_overlapping(pred_comms, ground_truth):
        nodes = {node for com in pred_comms + ground_truth for node in com}
        return len(nodes)

    def h(x):
        return -1 * x * (math.log(x) / math.log(2)) if x > 0 else 0

    def H_func(comm):
        p1 = len(comm) / overlapping_nodes
        p0 = 1 - p1
        return h(p0) + h(p1)

    def h_xi_joint_yj(xi, yj):
        p11 = get_intersection(xi, yj) / overlapping_nodes
        p10 = get_difference(xi, yj) / overlapping_nodes
        p01 = get_difference(yj, xi) / overlapping_nodes
        p00 = 1 - p11 - p10 - p01

        if h(p11) + h(p00) >= h(p01) + h(p10):
            return h(p11) + h(p10) + h(p01) + h(p00)
        return H_func(xi) + H_func(yj)

    def h_xi_given_yj(xi, yj):
        return h_xi_joint_yj(xi, yj) - H_func(yj)

    def H_XI_GIVEN_Y(xi, Y):
        res = h_xi_given_yj(xi, Y[0])
        for y in Y:
            res = min(res, h_xi_given_yj(xi, y))
        denom = H_func(xi)
        return res / denom if denom else 0

    def H_X_GIVEN_Y(X, Y):
        if not X:
            return 0
        res = 0
        for idx in range(len(X)):
            res += H_XI_GIVEN_Y(X[idx], Y)
        return res / len(X)

    if len(pred) == 0 or len(gt) == 0:
        return 0

    overlapping_nodes = get_overlapping(pred, gt)
    if overlapping_nodes == 0:
        return 0

    return 1 - 0.5 * (H_X_GIVEN_Y(pred, gt) + H_X_GIVEN_Y(gt, pred))

def evaluate_overlap_metrics(pred_comms, true_comms):
    if not pred_comms or not true_comms:
        print("[eval-overlap] skipped: empty communities for NMI computation")
        return None

    pred_scores = np.zeros((len(pred_comms), 4))
    truth_scores = np.zeros((len(true_comms), 4))

    for i, pred_comm in enumerate(pred_comms):
        np.max([compare_comm(pred_comm, true_comms[j]) for j in range(len(true_comms))], 0, out=pred_scores[i])

    for j, true_comm in enumerate(true_comms):
        np.max([compare_comm(pred_comms[i], true_comm) for i in range(len(pred_comms))], 0, out=truth_scores[j])
    truth_scores[:, :2] = truth_scores[:, [1, 0]]

    mean_score_all = (pred_scores.mean(0) + truth_scores.mean(0)) / 2.

    comm_nodes = {node for com in true_comms for node in com}
    pred_nodes = {node for com in pred_comms for node in com}
    percent = len(comm_nodes & pred_nodes) / len(comm_nodes) if comm_nodes else 0.0

    nmi_score = get_nmi_score(pred_comms, true_comms)

    print("[eval-overlap] AvgF1: {:.4f} AvgJaccard: {:.4f} NMI: {:.4f} Detect%: {:.4f}".format(
        mean_score_all[2], mean_score_all[3], nmi_score, percent
    ), flush=True)
    return round(mean_score_all[2], 4), round(mean_score_all[3], 4), round(nmi_score, 4)

def precompute_refine_assets(num_nodes, adj_mat, node_labels):
    print("Pre-computing global adjacency list and node labels...", flush=True)
    global_adj_list = {node: set(adj_mat[node].indices) for node in range(num_nodes)}
    if isinstance(node_labels, np.ndarray):
        node_labels_dict = {i: int(node_labels[i]) for i in range(len(node_labels))}
    elif isinstance(node_labels, (list, tuple)):
        node_labels_dict = {i: int(node_labels[i]) for i in range(len(node_labels))}
    else:
        node_labels_dict = {int(k): int(v) for k, v in node_labels.items()}
    print("Pre-computation completed", flush=True)
    return global_adj_list, node_labels_dict

def make_candidate_community(comm, global_adj_list, node_labels_dict, comm_idx=None):
    comm_set = set(comm)
    edges = []
    for node in comm:
        neighbors = global_adj_list.get(node, set()) & comm_set
        for neighbor in neighbors:
            if node < neighbor:
                edges.append((node, neighbor))
    result = {
        'nodes': comm,
        'edges': edges,
        'node_labels': {node: node_labels_dict[node] for node in comm}
    }
    if comm_idx is not None:
        result['id'] = comm_idx
    return result

def build_candidate_communities(sampled_comms, global_adj_list, node_labels_dict):
    print(f"Preparing {len(sampled_comms)} sampled communities...", flush=True)
    milestone = max(1, len(sampled_comms) // 10)
    candidate_communities = []
    for idx, comm in enumerate(sampled_comms):
        if (idx + 1) % milestone == 0 or idx == len(sampled_comms) - 1:
            print(f"Prepared {idx + 1}/{len(sampled_comms)} communities", flush=True)
        candidate_communities.append(make_candidate_community(comm, global_adj_list, node_labels_dict, idx))
    return candidate_communities

def extract_python_code(content):
    if content.startswith("```python"):
        return content.split("```python", 1)[1].rsplit("```", 1)[0].strip()
    if content.startswith("```"):
        return content.strip("`").strip()
    match = re.search(r'```python\s*(.*?)\s*```', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content.strip()

def compile_batch_refine_function(python_code):
    if not python_code:
        return None, None
    print("=== LLM Generated Python Code ===", flush=True)
    print(python_code)
    print("=== End Python Code ===", flush=True)
    exec_globals = {}
    exec(python_code, exec_globals)
    return exec_globals.get('batch_refine_communities'), python_code

def build_all_communities(pred_comms, global_adj_list, node_labels_dict, input_format):
    print(f"Preparing data for {len(pred_comms)} communities...", flush=True)
    all_communities_data = []
    comm_idx_mapping = {}
    for i, comm in enumerate(pred_comms):
        if len(comm) < 2:
            continue
        comm_set = set(comm)
        community_data = {
            'nodes': comm,
            'node_labels': {node: node_labels_dict[node] for node in comm}
        }
        if input_format == "degree":
            community_data['degree'] = {node: len(global_adj_list[node] & comm_set) for node in comm}
        else:
            community_data['adjacency_list'] = {node: list(global_adj_list[node] & comm_set) for node in comm}
        comm_idx_mapping[len(all_communities_data)] = i
        all_communities_data.append(community_data)
    print("Data preparation completed", flush=True)
    return all_communities_data, comm_idx_mapping

def run_refine_batches(batch_refine_function, batch_refine_code, all_communities_data, num_processes):
    MAX_BATCH_SIZE = 10000
    num_communities = len(all_communities_data)
    if num_communities == 0:
        return {}, None, None

    if batch_refine_code is None or num_processes <= 1:
        print("Warning: No parallel source available, using single-process refinement", flush=True)
        drop_nodes_mapping = {}
        num_batches = (num_communities + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
        for batch_idx in range(num_batches):
            start_idx = batch_idx * MAX_BATCH_SIZE
            end_idx = min(start_idx + MAX_BATCH_SIZE, num_communities)
            print(f"Processing batch {batch_idx + 1}/{num_batches} ({end_idx - start_idx} communities)...", flush=True)
            batch_result = batch_refine_function(all_communities_data[start_idx:end_idx])
            drop_nodes_mapping.update({start_idx + k: v for k, v in batch_result.items()})
        total_drop_nodes = sum(len(nodes) for nodes in drop_nodes_mapping.values())
        print(f"Total nodes to drop across all communities: {total_drop_nodes}", flush=True)
        return drop_nodes_mapping, None, None

    communities_per_process = max(1, (num_communities + num_processes - 1) // num_processes)
    actual_batch_size = min(communities_per_process, MAX_BATCH_SIZE)
    batches = []
    for start_idx in range(0, num_communities, actual_batch_size):
        end_idx = min(start_idx + actual_batch_size, num_communities)
        batches.append((start_idx, all_communities_data[start_idx:end_idx]))

    print(f"Using joblib with {num_processes} parallel processes to process {num_communities} communities", flush=True)
    print(f"Batch size limit: {MAX_BATCH_SIZE} communities per batch", flush=True)
    print(f"Split into {len(batches)} batches (~{actual_batch_size} communities per batch)", flush=True)

    def process_batch(batch):
        start_idx, batch_data = batch
        exec_globals = {
            'np': np,  # 添加numpy导入，供LLM生成的代码使用
            'math': math,  # 添加math导入
        }
        exec(batch_refine_code, exec_globals)
        local_fn = exec_globals['batch_refine_communities']
        batch_result = local_fn(batch_data)
        return {start_idx + idx: nodes for idx, nodes in batch_result.items() if isinstance(idx, int)}

    batch_results = Parallel(n_jobs=num_processes, verbose=5)(
        delayed(process_batch)(batch) for batch in batches
    )

    drop_nodes_mapping = {}
    for result in batch_results:
        drop_nodes_mapping.update(result)

    print(f"Successful batches: {len(batch_results)}/{len(batches)}", flush=True)
    total_drop_nodes = sum(len(nodes) for nodes in drop_nodes_mapping.values())
    print(f"Total nodes to drop across all communities: {total_drop_nodes}", flush=True)
    return drop_nodes_mapping

def run_merge_batches(ego_data_cache, adj_mat, args, merge_k_ego_networks_fn, 
                      center_nodes, jaccard_similarity):
    """
    单进程处理merge操作
    
    Args:
        ego_data_cache: dict mapping node_id -> (nodes, edges, labels)
        adj_mat: adjacency matrix (scipy sparse matrix)
        args: arguments object
        merge_k_ego_networks_fn: merge function from LLM
        center_nodes: list of all center node IDs to process
        jaccard_similarity: function to calculate Jaccard similarity
    
    Returns:
        candidate_communities: list of merged communities
        processed_as_center: set of processed center nodes
    """
    num_center_nodes = len(center_nodes)
    
    if num_center_nodes == 0:
        return [], set()
    
    print("Using single-process merge", flush=True)
    # 使用普通set
    processed_as_center = set()
    candidate_communities = []
    
    # 按照ego density对center nodes进行排序（从高到低）
    # EgoDensity(v) = (2 * |E_ego(v)|) / (|V_ego(v)| * (|V_ego(v)| - 1))
    print("Computing ego density for all center nodes and sorting...", flush=True)
    center_node_densities = []
    for center_node in center_nodes:
        center_k_ego_nodes, center_k_ego_edges, _ = ego_data_cache[center_node]
        num_nodes = len(center_k_ego_nodes)
        num_edges = len(center_k_ego_edges)
        
        if num_nodes <= 1:
            ego_density = 0.0
        else:
            # EgoDensity(v) = (2 * |E_ego(v)|) / (|V_ego(v)| * (|V_ego(v)| - 1))
            ego_density = (2.0 * num_edges) / (num_nodes * (num_nodes - 1))
        
        center_node_densities.append((center_node, ego_density))
    
    # 按照ego density从高到低排序
    center_node_densities.sort(key=lambda x: x[1], reverse=True)
    center_nodes = [node for node, _ in center_node_densities]
    print(f"Sorted {len(center_nodes)} center nodes by ego density (highest first)", flush=True)
    
    for center_node in center_nodes:
        if center_node in processed_as_center:
            continue
        
        center_k_ego_nodes, center_k_ego_edges, center_k_ego_labels = ego_data_cache[center_node]
        center_k_ego_set = set(center_k_ego_nodes)
        direct_neighbors = adj_mat[center_node].indices
        
        neighbor_similarities = []
        for neighbor_node in direct_neighbors:
            if neighbor_node in processed_as_center:
                continue
            neighbor_k_ego_nodes, _, _ = ego_data_cache[neighbor_node]
            neighbor_k_ego_set = set(neighbor_k_ego_nodes)
            jaccard_sim = jaccard_similarity(center_k_ego_set, neighbor_k_ego_set)
            neighbor_similarities.append((neighbor_node, jaccard_sim))
        
        neighbor_similarities.sort(key=lambda x: x[1], reverse=True)
        
        if args.topk_neighbors_merge > 0:
            selected_neighbors = [node for node, sim in neighbor_similarities[:args.topk_neighbors_merge]]
        else:
            selected_neighbors = [node for node, sim in neighbor_similarities]
        
        selected_neighbors = [n for n in selected_neighbors if n not in processed_as_center]
        
        if args.input_format == "degree":
            center_degrees = [len([n for n in center_k_ego_nodes 
                                  if (node, n) in center_k_ego_edges or (n, node) in center_k_ego_edges]) 
                            for node in center_k_ego_nodes]
        else:
            center_adj_list = {}
            for node in center_k_ego_nodes:
                center_adj_list[node] = [n for n in center_k_ego_nodes 
                                       if (node, n) in center_k_ego_edges or (n, node) in center_k_ego_edges]
        
        neighbor_k_ego_networks = []
        for neighbor_node in selected_neighbors:
            if neighbor_node in processed_as_center:
                continue
            neighbor_k_ego_nodes, neighbor_k_ego_edges, neighbor_k_ego_labels = ego_data_cache[neighbor_node]
            
            if args.input_format == "degree":
                neighbor_degrees = [len([n for n in neighbor_k_ego_nodes 
                                        if (node, n) in neighbor_k_ego_edges or (n, node) in neighbor_k_ego_edges]) 
                                  for node in neighbor_k_ego_nodes]
                neighbor_k_ego_networks.append((neighbor_node, list(neighbor_k_ego_nodes), neighbor_degrees, list(neighbor_k_ego_labels)))
            else:
                neighbor_adj_list_dict = {}
                for node in neighbor_k_ego_nodes:
                    neighbor_adj_list_dict[node] = [n for n in neighbor_k_ego_nodes 
                                                   if (node, n) in neighbor_k_ego_edges or (n, node) in neighbor_k_ego_edges]
                neighbor_k_ego_networks.append((neighbor_node, list(neighbor_k_ego_nodes), neighbor_adj_list_dict, list(neighbor_k_ego_labels)))
        
        if not neighbor_k_ego_networks:
            candidate_communities.append(list(center_k_ego_nodes))
            processed_as_center.add(center_node)
            continue
        
        center_k_ego_set_initial = set(center_k_ego_nodes)
        
        # 调用LLM生成的merge函数
        if args.input_format == "degree":
            merged_result = merge_k_ego_networks_fn(
                list(center_k_ego_nodes), center_degrees, list(center_k_ego_labels),
                neighbor_k_ego_networks, current_community=None
            )
        else:
            merged_result = merge_k_ego_networks_fn(
                list(center_k_ego_nodes), center_adj_list, list(center_k_ego_labels),
                neighbor_k_ego_networks, current_community=None
            )
        
        merged_set = set(merged_result)
        merged_neighbors = []
        for neighbor_tuple in neighbor_k_ego_networks:
            neighbor_node = neighbor_tuple[0]
            neighbor_k_ego_nodes = neighbor_tuple[1]
            neighbor_set = set(neighbor_k_ego_nodes)
            if neighbor_set & merged_set and not (neighbor_set <= center_k_ego_set_initial):
                merged_neighbors.append(neighbor_node)
        
        processed_as_center.add(center_node)
        
        for merged_neighbor_node in merged_neighbors:
            merged_neighbor_k_ego_nodes, _, _ = ego_data_cache[merged_neighbor_node]
            processed_as_center.add(merged_neighbor_node)
            processed_as_center.update(merged_neighbor_k_ego_nodes)

            candidate_communities.append(list(merged_neighbor_k_ego_nodes))
        
        candidate_communities.append(list(merged_result))
    
    return candidate_communities, processed_as_center


def apply_drop_nodes(pred_comms, comm_idx_mapping, drop_nodes_mapping):
    refined_comms = []
    total_kept = 0
    total_removed = 0
    comm_idx_to_data_idx = {pred_idx: data_idx for data_idx, pred_idx in comm_idx_mapping.items()}
    for i, comm in enumerate(pred_comms):
        if len(comm) < 2:
            refined_comms.append(comm)
            total_kept += len(comm)
            continue
        data_idx = comm_idx_to_data_idx.get(i)
        if data_idx is None:
            refined_comms.append(comm)
            total_kept += len(comm)
            continue
        drop_nodes = drop_nodes_mapping.get(data_idx, [])
        drop_nodes_set = set(drop_nodes)
        kept_nodes = [node for node in comm if node not in drop_nodes_set]
        total_kept += len(kept_nodes)
        total_removed += len(drop_nodes)
        if len(kept_nodes) >= 2:
            refined_comms.append(kept_nodes)
        else:
            print(f"Community {i} too small after refinement ({len(kept_nodes)} nodes), discarding", flush=True)
    return refined_comms, total_kept, total_removed

def majority_label(nodes_list):
    if len(nodes_list) == 0:
        return 0
    labels = [int(node_labels[i]) for i in nodes_list]
    counts = {}
    for lb in labels:
        counts[lb] = counts.get(lb, 0) + 1
    return max(counts.items(), key=lambda x: x[1])[0]

def compute_community_metrics(community_nodes, global_adj_list, total_volume=None):
    """
    一次性计算社区的多项指标（优化：避免重复遍历边）
    
    在一次遍历中同时计算：
    - internal_edges: 社区内部边数 |E(S)|
    - cut_C: 边界边数（连接社区内部和外部节点的边数）
    - vol_C: 社区内所有节点的度数之和
    
    Args:
        community_nodes: 社区节点集合（set 或 list）
        global_adj_list: 全局邻接表，dict[node_id] -> set/list of neighbors
        total_volume: 图的总度数（可选，如果提供则计算 vol(V\C)）
    
    Returns:
        tuple: (internal_edges, cut_C, vol_C, vol_V_minus_C)
               - internal_edges: 内部边数
               - cut_C: 边界边数
               - vol_C: 社区度数之和
               - vol_V_minus_C: 图的总度数 - vol_C（如果 total_volume 提供）
    """
    if len(community_nodes) == 0:
        return (0, 0, 0, total_volume if total_volume is not None else None)
    
    comm_set = set(community_nodes)
    internal_edges = 0
    vol_C = 0
    cut_C = 0
    
    for node in comm_set:
        if node not in global_adj_list:
            continue
        
        neighbors = global_adj_list[node]
        node_degree = len(neighbors) if isinstance(neighbors, (set, list)) else 0
        vol_C += node_degree
        
        for neighbor in neighbors:
            if neighbor in comm_set:
                if node < neighbor:
                    internal_edges += 1
            else:
                # 边界边
                cut_C += 1
    
    vol_V_minus_C = (total_volume - vol_C) if total_volume is not None else None
    return (internal_edges, cut_C, vol_C, vol_V_minus_C)

def check_anti_hollow_constraint(community_nodes, global_adj_list, epsilon=0.3, metrics_cache=None):
    """
    Stage 1: 轻量级硬约束过滤 (Anti-Hollow Filtering)
    
    检查社区是否满足 spanning tree redundancy constraint:
    |E(S)| ≥ (1 + ε)(|S| - 1)
    
    其中：
    - |E(S)|: 社区内部的边数
    - |S|: 社区节点数
    - ε: 参数，范围 [0.2, 0.5]，默认 0.3
    
    该约束确保社区内部的边数显著多于最小连通结构（生成树）的边数，
    从而有效过滤掉树状、链状等空心结构。
    
    Args:
        community_nodes: 社区节点集合（set 或 list）
        global_adj_list: 全局邻接表，dict[node_id] -> set/list of neighbors
        epsilon: 约束参数，范围 [0.2, 0.5]，默认 0.3
        metrics_cache: 可选的预计算指标缓存 (internal_edges, ...)，避免重复计算
    
    Returns:
        bool: True 表示满足约束（通过过滤），False 表示不满足（应丢弃）
    """
    if len(community_nodes) <= 1:
        # 单节点或空社区，不满足约束（因为 |S| - 1 = 0）
        return False
    
    # 确保 epsilon 在有效范围内
    epsilon = max(0.2, min(0.5, epsilon))
    
    S_size = len(community_nodes)
    
    if metrics_cache is not None:
        E_S = metrics_cache[0]  # internal_edges
    else:
        comm_set = set(community_nodes)
        E_S = 0
        for node in comm_set:
            if node not in global_adj_list:
                continue
            neighbors = global_adj_list[node]
            for neighbor in neighbors:
                if neighbor in comm_set and node < neighbor:
                    E_S += 1
    
    threshold = (1 + epsilon) * (S_size - 1)
    return E_S >= threshold

def compute_local_conductance(community_nodes, global_adj_list, total_volume, metrics_cache=None):
    """
    计算社区的 local conductance 指标
    
    Local Conductance 定义：
    conductance(C) = cut(C) / min(vol(C), vol(V\C))
    
    其中：
    - cut(C): 社区 C 的边界边数（连接 C 内部节点和外部节点的边数）
    - vol(C): 社区 C 中所有节点的度数之和
    - vol(V\C): 图的总度数 - vol(C)
    
    Args:
        community_nodes: 社区节点集合（set 或 list）
        global_adj_list: 全局邻接表，dict[node_id] -> set/list of neighbors
        total_volume: 图的总度数（所有节点的度数之和）
        metrics_cache: 可选的预计算指标缓存 (internal_edges, cut_C, vol_C, vol_V_minus_C)，避免重复计算
    
    Returns:
        float: local conductance 值，越小表示社区越紧密（质量越好）
    """
    if len(community_nodes) == 0:
        return float('inf')  # 空社区返回无穷大
    
    # 如果提供了缓存，直接使用；否则计算
    if metrics_cache is not None:
        cut_C, vol_C, vol_V_minus_C = metrics_cache[1], metrics_cache[2], metrics_cache[3]
    else:
        _, cut_C, vol_C, vol_V_minus_C = compute_community_metrics(community_nodes, global_adj_list, total_volume)
    
    # 避免除零错误
    min_vol = min(vol_C, vol_V_minus_C)
    if min_vol == 0:
        return float('inf')
    
    conductance = cut_C / min_vol
    return conductance

def rank_by_density_weighted_conductance(communities_as_sets, global_adj_list, num_edges, epsilon=0.3, delta=1e-6):
    """
    使用 Density-Weighted Conductance (结构增益因子) 方法对社区进行排序
    
    公式：
    Score(S) = Cut(S, S̄) / (|E(S)| – ε(|S| – 1))
    
    其中：
    - Cut(S, S̄): 社区S的边界边数（连接社区内部节点和外部节点的边数）
    - |E(S)|: 社区S内部的边数
    - ε: 参数，控制最小连通结构的权重
    - |S|: 社区S的节点数
    - (|S| – 1): 最小连通结构所需的边数
    
    融合逻辑：
    - 将分母从单纯的 Vol(S) 替换为"有效盈余边数"
    - 如果内部边数接近临界点，分母变小，得分变差（数值变大）
    - 如果内部边数远超阈值，分母变大，会显著压低 Conductance，使该社区在评估中脱颖而出
    
    注意：为防止分母为负，实际计算时使用 max(δ, |E(S)| – ε(|S| - 1))，其中 δ 是一个极小正数
    
    Args:
        communities_as_sets: 社区列表，每个社区是 set 或 list
        global_adj_list: 全局邻接表
        num_edges: 图的边数（用于计算总度数）
        epsilon: 参数，控制最小连通结构的权重，默认 0.3
        delta: 极小正数，用于防止分母为负，默认 1e-6
    
    Returns:
        ranked_communities: 按 Score 从小到大排序的社区列表（Score 越小越好）
        scores: dict[frozenset(community)] -> Score 值
    """
    print("=" * 80, flush=True)
    print("Ranking communities with Density-Weighted Conductance (结构增益因子)", flush=True)
    print("=" * 80, flush=True)
    print(f"Formula: Score(S) = Cut(S, S̄) / max(δ, |E(S)| – ε(|S| – 1))", flush=True)
    print(f"Parameters: ε = {epsilon}, δ = {delta}", flush=True)
    
    # 计算图的总度数（所有节点的度数之和）
    total_volume = sum(len(neighbors) if isinstance(neighbors, (set, list)) else 0 
                      for neighbors in global_adj_list.values())
    
    # 计算每个社区的 Density-Weighted Conductance 分数
    scores = {}
    for comm in communities_as_sets:
        comm_set = set(comm) if not isinstance(comm, set) else comm
        comm_frozen = frozenset(comm_set)
        
        # 计算社区指标
        metrics = compute_community_metrics(comm_set, global_adj_list, total_volume)
        internal_edges = metrics[0]  # |E(S)|
        cut_S = metrics[1]  # Cut(S, S̄)
        S_size = len(comm_set)  # |S|
        
        # 计算分母：max(δ, |E(S)| – ε(|S| – 1))
        # 其中 |E(S)| – ε(|S| – 1) 表示"有效盈余边数"
        min_connected_edges = epsilon * (S_size - 1)  # ε(|S| – 1)
        effective_surplus_edges = internal_edges - min_connected_edges  # |E(S)| – ε(|S| – 1)
        denominator = max(delta, effective_surplus_edges)
        
        # 计算 Score(S) = Cut(S, S̄) / max(δ, |E(S)| – ε(|S| – 1))
        if denominator > 0:
            score = cut_S / denominator
        else:
            # 理论上不会发生（因为使用了 max(δ, ...)），但为了安全起见
            score = float('inf')
        
        scores[comm_frozen] = score
    
    # 按 Score 从小到大排序（Score 越小，社区质量越好）
    print("Sorting communities by Density-Weighted Conductance (ascending order, lower is better)...", flush=True)
    ranked_communities = sorted(communities_as_sets, 
                                key=lambda comm: scores.get(frozenset(comm), float('inf')))
    
    if len(scores) > 0:
        valid_scores = [s for s in scores.values() if s != float('inf')]
        if valid_scores:
            print(f"Ranking completed. Score range: [{min(valid_scores):.6f}, {max(valid_scores):.6f}]", flush=True)
        else:
            print("Ranking completed. All scores are infinite.", flush=True)
    else:
        print("Ranking completed. No communities to rank.", flush=True)
    
    print("=" * 80, flush=True)
    
    return ranked_communities, scores

def rank_by_local_conductance(communities_as_sets, global_adj_list, num_edges, epsilon=0.3):
    """
    使用两阶段方法对社区进行排序：
    Stage 1: 轻量级硬约束过滤 (Anti-Hollow Filtering) - 过滤空心社区
    Stage 2: Local Conductance 计算 (Primary Scoring) - 按 conductance 排序
    
    Args:
        communities_as_sets: 社区列表，每个社区是 set 或 list
        global_adj_list: 全局邻接表
        num_edges: 图的边数（用于计算总度数）
        epsilon: Anti-Hollow Filtering 参数，范围 [0.2, 0.5]，默认 0.3
    
    Returns:
        ranked_communities: 按 conductance 从小到大排序的社区列表（conductance 越小越好）
        conductance_scores: dict[frozenset(community)] -> conductance 值
    """
    print("=" * 80, flush=True)
    print("Stage 1: Lightweight Hard Constraint Filtering (Anti-Hollow Filtering)", flush=True)
    print("=" * 80, flush=True)
    print(f"Filtering communities with constraint: |E(S)| ≥ (1 + ε)(|S| - 1), where ε = {epsilon}", flush=True)
    
    # 计算图的总度数（所有节点的度数之和）
    total_volume = sum(len(neighbors) if isinstance(neighbors, (set, list)) else 0 
                      for neighbors in global_adj_list.values())
    
    filtered_communities = []
    filtered_metrics_cache = {}  # 缓存通过过滤的社区的指标
    discarded_count = 0
    
    for comm in communities_as_sets:
        comm_set = set(comm) if not isinstance(comm, set) else comm
        comm_frozen = frozenset(comm_set)
        
        # 一次性计算所有指标（内部边数、边界边数、vol等）
        metrics = compute_community_metrics(comm_set, global_adj_list, total_volume)
        internal_edges = metrics[0]
        
        S_size = len(comm_set)
        epsilon_clamped = max(0.2, min(0.5, epsilon))
        threshold = (1 + epsilon_clamped) * (S_size - 1)
        
        if S_size > 1 and internal_edges >= threshold:
            filtered_communities.append(comm)
            filtered_metrics_cache[comm_frozen] = metrics
        else:
            discarded_count += 1
    
    print(f"Stage 1 completed: {len(filtered_communities)} communities passed, {discarded_count} communities discarded", flush=True)
    
    if len(filtered_communities) == 0:
        print("Warning: No communities passed Stage 1 filtering!", flush=True)
        return [], {}
    
    print("\n" + "=" * 80, flush=True)
    print("Stage 2: Local Conductance Calculation (Primary Scoring)", flush=True)
    print("=" * 80, flush=True)
    print("Computing local conductance for filtered communities (reusing Stage 1 metrics)...", flush=True)
    
    # Stage 2: 使用预计算的指标计算 conductance（避免重复遍历边）
    conductance_scores = {}
    for comm in filtered_communities:
        comm_set = set(comm) if not isinstance(comm, set) else comm
        comm_frozen = frozenset(comm_set)
        metrics_cache = filtered_metrics_cache[comm_frozen]
        conductance = compute_local_conductance(comm_set, global_adj_list, total_volume, metrics_cache)
        conductance_scores[comm_frozen] = conductance
    
    # 按 conductance 从小到大排序（conductance 越小，社区质量越好）
    print("Sorting communities by local conductance (ascending order, lower is better)...", flush=True)
    ranked_communities = sorted(filtered_communities, 
                                key=lambda comm: conductance_scores.get(frozenset(comm), float('inf')))
    
    if len(conductance_scores) > 0:
        print(f"Stage 2 completed. Conductance range: [{min(conductance_scores.values()):.6f}, {max(conductance_scores.values()):.6f}]", flush=True)
    else:
        print("Stage 2 completed. No communities to rank.", flush=True)
    
    print("=" * 80, flush=True)
    
    return ranked_communities, conductance_scores

if __name__ == '__main__':
    print(os.getcwd())
    print('= ' * 20)
    print('= ' * 20)

    parser = argparse.ArgumentParser(description="K-EGO Community Detection")

    # Dataset choices
    parser.add_argument("--dataset", type=str, default="facebook", 
                       help="Dataset name: ['amazon', 'dblp', 'lj', 'facebook', 'twitter']")
    parser.add_argument("--data_root", type=str, default="./data",
                       help="Root directory for data files")
    parser.add_argument("--device", type=str, default="cuda:1")

    # K-EGO related parameters
    parser.add_argument("--k", type=int, default=2, 
                       help="Number of hops for k-ego network extraction")
    # parser.add_argument("--num_pred", type=int, default=1000, 
    #                    help="Number of predicted communities")
    # parser.add_argument("--run_times", type=int, default=10, 
    #                    help="Number of runs for evaluation")
    parser.add_argument("--n_roles", type=int, default=4,
                       help="Number of node labels for k-means in label_nodes")
    # parser.add_argument("--max_neighbor_checks", type=int, default=5,
    #                    help="Max neighbors to query per node for LLM merge decision")
    
    # LLM sampling parameters
    parser.add_argument("--merge_sample_size", type=int, default=3,
                       help="Number of k-ego samples to prepare for LLM merge. If 1, represents one node's k-ego and its direct neighbors' k-ego")
    parser.add_argument("--refine_sample_size", type=int, default=10,
                       help="Number of communities to sample for node refinement rule learning")
    parser.add_argument("--refine_jobs", type=int, default=4,
                       help="Number of parallel processes for community refinement processing")
    
    # Input format parameter
    parser.add_argument("--input_format", type=str, default="degree", choices=["degree", "adjacency_list"],
                       help="Input format for LLM rules: 'degree' for node degrees, 'adjacency_list' for adjacency lists")
    
    # Neighbor sampling parameter
    parser.add_argument("--max_neighbors_per_center", type=int, default=3,
                       help="Maximum number of direct neighbors to include per center node. If -1, include all neighbors. Default: 3")

    # Merge neighbor selection parameter
    parser.add_argument("--topk_neighbors_merge", type=int, default=3,
                       help="Number of top-k similar neighbors to use for merge based on Jaccard similarity. If -1, use all neighbors. Default: 3")
    
    # Ranking method parameter
    parser.add_argument("--ranking_method", type=str, default="conductance", choices=["conductance"],
                       help="Ranking method to use: 'conductance' for Density-Weighted Conductance (lower is better). Default: 'conductance'")
    
    # Anti-Hollow Filtering parameter (for conductance ranking)
    parser.add_argument("--epsilon", type=float, default=0.3,
                       help="Epsilon parameter for Density-Weighted Conductance: controls weight of minimum connected structure in denominator. Used in formula: Score(S) = Cut(S, S̄) / max(δ, |E(S)| – ε(|S| – 1)). Default: 0.3")
    
    # Tree depth range parameter
    parser.add_argument("--tree_depth_range", type=int, nargs=2, default=[8, 12],
                       metavar=('MIN', 'MAX'),
                       help="Range of rule nodes for decision tree: [MIN, MAX]. Default: [8, 12]")
    
    # ablation study mode
    parser.add_argument("--ablation_mode", type=str, default="full", 
                   choices=["full", "kego", "no_merge", "no_refine", "no_selection"],
                   help="Mode for ablation study")

    args = parser.parse_args()
    
    # Convert tree_depth_range list to tuple
    args.tree_depth_range = tuple(args.tree_depth_range)

    if args.dataset == "facebook":
        args.num_pred = 200
    elif args.dataset == "lj":
        args.num_pred = 1000
    elif args.dataset == "amazon": 
        args.num_pred = 1500
    elif args.dataset in ["dblp","twitter"]:
        args.num_pred = 5000

    print(f"Dataset: {args.dataset}")
    print(f"Input format: {args.input_format}")
    print(f"K-EGO hops: {args.k}")
    print(f"Number of predictions per run: {args.num_pred}")
    # print(f"Number of evaluation runs: {args.run_times}")
    print("\n")
    ##########################################################
    ################### Stage 1: Initialize ###################
    ##########################################################
    print("Loading data...")
    adj_mat, comms, _ ,num_nodes,num_edges = load_snap_dataset(args.dataset, root=args.data_root)
    print(f"Nodes: {num_nodes}, Edges: {num_edges}, Communities: {len(comms)}")
    print("Node Labeling.....")
    node_neighbors = {u: set(adj_mat[u].indices) for u in range(num_nodes)}
    node_labels, _ = label_nodes(node_neighbors, args.n_roles)
    print("complete node labeling!!!", flush=True)

    ##########################################################
    ################### Stage 1: Initialize (cont.) ##########
    ##########################################################
    print("Extracting K-EGO networks for all nodes...", flush=True)

    kego_file_path = os.path.join(args.data_root, args.dataset, f"{args.k}-ego.txt")
    node2ego_mapping = []
    
    if os.path.exists(kego_file_path):
        print(f"Loading pre-computed k-ego networks from {kego_file_path}")
        with open(kego_file_path, 'r') as file:
            for line in file.readlines():
                content = line.split(" ")[1:]
                node2ego_mapping.append([int(node) for node in content if node.strip()])
    else:
        print(f"Computing k-ego networks (k={args.k}) for {num_nodes} nodes...")
        ei_rows, ei_cols = adj_mat.nonzero()
        edge_index = torch.tensor(np.vstack((ei_rows, ei_cols)), dtype=torch.long)
        for node in range(num_nodes):
            node_k_ego, _, _, _ = k_hop_subgraph(
                node, 
                num_hops=args.k, 
                edge_index=edge_index,
                num_nodes=num_nodes
            )
            node_k_ego = node_k_ego.detach().cpu().numpy().tolist()
            node2ego_mapping.append(node_k_ego)
            
            if node % 5000 == 0:
                print(f"Processed {node}/{num_nodes} nodes...", flush=True)
        
        # 保存k-ego网络到文件以便下次使用
        os.makedirs(os.path.dirname(kego_file_path), exist_ok=True)
        with open(kego_file_path, 'w') as file:
            for i, ego_net in enumerate(node2ego_mapping):
                file.write(f"{i} {' '.join(map(str, ego_net))}\n")
        print(f"Saved k-ego networks to {kego_file_path}")
    
        print("K-EGO extraction completed", flush=True)
    print(f"Generated {len(node2ego_mapping)} k-ego networks", flush=True)
    
    ##########################################################
    ################### Stage 2: Merge #######################
    ##########################################################

    # # print("Batch merge community with LLM (15 subgraph pairs)...", flush=True)
    #openai原生api
    client = OpenAI(api_key=API_KEY)

    # Pre-compute and cache k-ego data (nodes, edges, labels) for all nodes to avoid processing twice
    print("Pre-computing k-ego networks data (nodes, edges, labels) for all nodes...", flush=True)
    ego_data_cache = {}  # node_id -> (nodes, edges, labels)
    
    for node_id in range(num_nodes):
        ego_nodes = node2ego_mapping[node_id]
        ego_node_set = set(ego_nodes)
        
        # Compute edges within k-ego network
        ego_edges = []
        for node in ego_nodes:
            if 0 <= node < adj_mat.shape[0]:
                node_neighbors = adj_mat[node].indices
                for neighbor in node_neighbors:
                    if neighbor in ego_node_set and node < neighbor:  # Avoid duplicate edges
                        ego_edges.append((node, neighbor))
        
        # Compute labels for each node in k-ego network
        ego_labels = [int(node_labels[node]) for node in ego_nodes]
        
        ego_data_cache[node_id] = (ego_nodes, ego_edges, ego_labels)
    
    print("Pre-computation completed", flush=True)
    
    # Prepare k_ego_samples for merge_prompt_v2
    random.seed(42)  # For reproducibility
    
    k_ego_samples = []
    
    # Determine how many neighbors to sample per center node
    max_neighbors = args.max_neighbors_per_center if args.max_neighbors_per_center > 0 else None  # None means all neighbors
    
    # Sample center nodes (unified handling for all sample sizes)
    sample_count = min(args.merge_sample_size, num_nodes)
    sampled_nodes = random.sample(range(num_nodes), sample_count)
    
    for center_node in sampled_nodes:
        center_nodes, center_edges, center_labels = ego_data_cache[center_node]
        
        # Get neighbors of center node
        neighbors = adj_mat[center_node].indices
        
        # Sample neighbors if max_neighbors is specified
        if max_neighbors is not None and len(neighbors) > max_neighbors:
            neighbors = random.sample(list(neighbors), max_neighbors)
        
        neighbors_list = []
        for neighbor_node in neighbors:
            neighbor_nodes, neighbor_edges, neighbor_labels = ego_data_cache[neighbor_node]
            neighbors_list.append((neighbor_node, neighbor_nodes, neighbor_edges, neighbor_labels))
        
        # Skip if no neighbors
        if not neighbors_list:
            continue
        
        k_ego_samples.append((center_node, center_nodes, center_edges, center_labels, neighbors_list))
    
    neighbor_info = f"max {max_neighbors} neighbors" if max_neighbors is not None else "all neighbors"
    print(f"Prepared {len(k_ego_samples)} k-ego samples for LLM ({neighbor_info} per center node)", flush=True)
    
    # Call merge_prompt_v2
    messages = merge_prompt_v2(
        args.dataset,
        num_nodes,
        num_edges,
        len(comms),
        k_ego_samples,
        args.input_format,
        args.tree_depth_range,
        args.data_root
    )
    
    print("Calling LLM for merge function generation...", flush=True)
    
    completion = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages
    )
    print("=" * 50)
    print(f"LLM response: {completion}")
    print("=" * 50)
# 提取内容
    content = completion.choices[0].message.content
    print("LLM output received", flush=True)
    
    # Parse LLM response
    if content.startswith("```"):
        content = content.strip("`").replace("json", "", 1).strip()
    LLM_response = json.loads(content)
    
    print("=== Output LLM Response ===", flush=True)
    print(json.dumps(LLM_response, indent=2, ensure_ascii=False))
    print("=== End LLM Output ===", flush=True)
    
    merge_execution_function = LLM_response.get('merge_execution_function', {})
    
    if args.input_format == "degree":
        merge_execution_code = merge_execution_function.get('code_degree', merge_execution_function.get('code', ''))
    else:  # adjacency_list
        merge_execution_code = merge_execution_function.get('code_adjacency_list', merge_execution_function.get('code', ''))
    
    merge_decision_tree = LLM_response.get('merge_decision_tree', {})
    
    if not merge_execution_code:
        raise ValueError(f"Missing required functions in LLM response. merge_execution_code: {bool(merge_execution_code)}")
    
    # Compile merge execution function
    print("Compiling merge execution function...", flush=True)
    exec_globals = {}
    exec(merge_execution_code, exec_globals)
    merge_k_ego_networks_fn = exec_globals.get('merge_k_ego_networks')
    if not merge_k_ego_networks_fn:
        raise ValueError("Failed to extract merge_k_ego_networks function")
    
    print("Using all nodes as center nodes...", flush=True)
    center_nodes = list(range(num_nodes))
    print(f"Using {len(center_nodes)} nodes as center nodes", flush=True)
    
    print("Applying merge function to merge communities...", flush=True)
    
    candidate_communities, processed_as_center = run_merge_batches(
        ego_data_cache=ego_data_cache,
        adj_mat=adj_mat,
        args=args,
        merge_k_ego_networks_fn=merge_k_ego_networks_fn,
        center_nodes=center_nodes,
        jaccard_similarity=jaccard_similarity
    )
    
    print("Merge process completed", flush=True)
    print(f"Generated {len(candidate_communities)} candidate communities from {len(processed_as_center)} processed center nodes (out of {len(center_nodes)} total nodes)", flush=True)
    
    pred_comms = candidate_communities
    
    print("Complete k-ego merging!!", flush=True)
    

    print(f"Generated {len(pred_comms)} candidate communities after rule application", flush=True)
    avg_size = sum(len(comm) for comm in pred_comms) / len(pred_comms) if pred_comms else 0
    print(f"Average size of candidate communities: {avg_size:.2f}", flush=True)

    evaluate_overlap_metrics(pred_comms, comms)
    print("=" * 50)


    
    ##########################################################
    ################### Stage 3: Refine ######################
    ##########################################################
    print("=" * 50)
    print("Refining community boundaries with LLM...", flush=True)
    print("=" * 50)

    global_adj_list, node_labels_dict = precompute_refine_assets(num_nodes, adj_mat, node_labels)

    large_comms = [comm for comm in pred_comms if len(comm) >= 3]
    print(f"Found {len(large_comms)} communities suitable for refinement (size >= 3)", flush=True)

    if not large_comms:
        print("No communities suitable for refinement, skipping node-level refinement", flush=True)
        refined_comms = pred_comms
    else:
        sample_size = min(args.refine_sample_size, len(large_comms))
        sampled_comms = random.sample(large_comms, sample_size)
        print(f"Sampling {sample_size} communities to generate node refinement function", flush=True)

        candidate_communities = build_candidate_communities(sampled_comms, global_adj_list, node_labels_dict)

        print(f"Calling LLM once with {len(candidate_communities)} communities to generate node refinement function", flush=True)
        messages = refine_prompt(candidate_communities, args.dataset, args.input_format, args.data_root)
        completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages
        )
        print("=" * 50)
        print(f"LLM response: {completion}")
        print("=" * 50)
        content = completion.choices[0].message.content
        python_code = extract_python_code(content)
        batch_refine_function, batch_refine_code = compile_batch_refine_function(python_code)

        if batch_refine_function is None:
            print("Failed to obtain batch refinement function from LLM response, keeping all communities unchanged", flush=True)
            refined_comms = pred_comms
        else:
            print("LLM refinement function generated", flush=True)
            print(f"Applying batch refinement function to all {len(pred_comms)} communities", flush=True)
            all_communities_data, comm_idx_mapping = build_all_communities(pred_comms, global_adj_list, node_labels_dict, args.input_format)
            print(f"Calling batch_refine_communities function with {len(all_communities_data)} communities", flush=True)
            drop_nodes_mapping = run_refine_batches(
                batch_refine_function, batch_refine_code, all_communities_data, args.refine_jobs
            )

            print("Applying drop decisions to communities...", flush=True)
            refined_comms, total_kept, total_removed = apply_drop_nodes(pred_comms, comm_idx_mapping, drop_nodes_mapping)
            print("Drop application completed", flush=True)

            print(f"Node refinement completed. Kept {total_kept} nodes, removed {total_removed} nodes", flush=True)
            print(f"Final communities: {len(refined_comms)} (from {len(pred_comms)} original)", flush=True)

    print("=" * 50)
    print("EVALUATION RESULTS:")
    evaluate(refined_comms, comms)
    evaluate_overlap_metrics(refined_comms, comms)
    print("=" * 50)


    
    ##########################################################
    ################### Stage 4: Rank ########################
    ##########################################################
    print("=" * 50)
    ranking_method_display = "Density-Weighted Conductance (结构增益因子)"
    print(f"Ranking communities with {ranking_method_display} method...", flush=True)
    print("=" * 50)
    
    if 'global_adj_list' not in locals():
        print("Precomputing adjacency list for ranking...", flush=True)
        global_adj_list, node_labels_dict = precompute_refine_assets(num_nodes, adj_mat, node_labels)
    else:
        print("Using precomputed adjacency list for ranking (optimized)...", flush=True)

    communities_as_sets_raw = [set(comm) for comm in refined_comms if len(comm) >= 2]
    print(f"Before deduplication: {len(communities_as_sets_raw)} communities (size >= 2)", flush=True)
    
    seen_communities = set()
    unique_communities_as_sets = []
    for comm in communities_as_sets_raw:
        comm_frozen = frozenset(comm) if isinstance(comm, (set, list)) else comm
        if comm_frozen not in seen_communities:
            seen_communities.add(comm_frozen)
            unique_communities_as_sets.append(comm)
    
    communities_as_sets = unique_communities_as_sets
    duplicates_removed = len(communities_as_sets_raw) - len(communities_as_sets)
    print(f"After deduplication: {len(communities_as_sets)} unique communities (removed {duplicates_removed} duplicates)", flush=True)
    
    # Build filename prefix with hyperparameters
    # Example: amazon_gpt5_2_4_20_5000_adj_conductance
    model_name_short = MODEL_NAME.replace("openai/", "").replace("deepseek", "ds").replace("-chat", "").replace("-", "").lower()
    input_format_short = "adj" if args.input_format == "adjacency_list" else "deg"
    ranking_method_short = "cond" if args.ranking_method == "conductance" else "dens"
    filename_prefix = f"{args.dataset}_{model_name_short}_{args.k}_{args.merge_sample_size}_{args.refine_sample_size}_{args.num_pred}_{input_format_short}_{ranking_method_short}"
    
    # Save all candidate communities before ranking
    output_comm_dir = "output_comm"
    os.makedirs(output_comm_dir, exist_ok=True)
    before_ranking_file = os.path.join(output_comm_dir, f"{filename_prefix}_before_ranking.txt")
    print(f"Saving all {len(communities_as_sets)} candidate communities before ranking to {before_ranking_file}...", flush=True)
    with open(before_ranking_file, 'w') as f:
        for comm in communities_as_sets:
            comm_list = list(comm) if isinstance(comm, (set, frozenset)) else comm
            f.write('\t'.join(map(str, comm_list)) + '\n')
    print(f"Saved {len(communities_as_sets)} communities before ranking", flush=True)
    
    if len(communities_as_sets) > 0:
        top_n = min(args.num_pred, len(communities_as_sets))
        
        print(f"Dataset: {args.dataset}, Top-N: {top_n} (min of args.num_pred={args.num_pred} and unique communities={len(communities_as_sets)})", flush=True)

        print("Performing ranking by Density-Weighted Conductance (结构增益因子)...", flush=True)
        ranked_communities, scores = rank_by_density_weighted_conductance(
            communities_as_sets,
            global_adj_list,
            num_edges,
            epsilon=args.epsilon
        )
        score_name = "Density_Weighted_Conductance"
        
        # 创建社区到分数的映射，用于后续保存和查询
        comm_to_score = {}
        for comm in ranked_communities:
            comm_frozen = frozenset(comm) if isinstance(comm, (set, list)) else comm
            score = scores.get(comm_frozen, float('inf'))
            comm_to_score[comm_frozen] = score
        top_n_communities = ranked_communities[:top_n]
        print(f"\nSelected top {len(top_n_communities)} communities from {len(ranked_communities)} ranked communities", flush=True)
        
        if len(top_n_communities) > 0:
            avg_community_size = sum(len(comm) for comm in top_n_communities) / len(top_n_communities)
            print(f"Average community size in top-{len(top_n_communities)} ranked communities: {avg_community_size:.2f}", flush=True)
        else:
            print("No communities in top-N results", flush=True)
        
        save_count = min(5000, len(ranked_communities))
        ranked_communities_file = os.path.join(output_comm_dir, f"{filename_prefix}_ranked_communities.txt")
        scores_file = os.path.join(output_comm_dir, f"{filename_prefix}_scores.txt")
        
        print(f"Saving top {save_count} ranked communities to {ranked_communities_file}...", flush=True)
        with open(ranked_communities_file, 'w') as f:
            for i, comm in enumerate(ranked_communities[:save_count]):
                comm_list = list(comm) if isinstance(comm, (set, frozenset)) else comm
                f.write('\t'.join(map(str, comm_list)) + '\n')
        
        print(f"Saving ranking scores to {scores_file}...", flush=True)
        with open(scores_file, 'w') as f:
            f.write(f"Rank\tCommunity_Size\t{score_name}\n")
            for i, comm in enumerate(ranked_communities[:save_count]):
                comm_list = list(comm) if isinstance(comm, (set, frozenset)) else comm
                comm_frozen = frozenset(comm) if isinstance(comm, (set, list)) else comm
                default_score = float('inf')
                score = comm_to_score.get(comm_frozen, default_score)
                f.write(f"{i+1}\t{len(comm_list)}\t{score:.6f}\n")
        
        print(f"Saved ranking results to {output_comm_dir}/ with prefix: {filename_prefix}", flush=True)
        
        print(f"Saved {save_count} ranked communities and scores to {output_comm_dir}/", flush=True)
        
        pred_comms = [list(comm) for comm in top_n_communities]
        
        print("Ranking completed", flush=True)
        print(f"Final communities after ranking: {len(pred_comms)}", flush=True)
    else:
        print("No communities to rank, keeping refined communities", flush=True)
        pred_comms = refined_comms
    
    print("=" * 50)

    ##########################################################
    ################### Evaluation ###########################
    ##########################################################

    print(f'-> (All)  # Comms: {len(pred_comms)}')
    print("=" * 50)
    print("EVALUATION RESULTS:")
    evaluate(pred_comms, comms)
    evaluate_overlap_metrics(pred_comms, comms)
    print("=" * 50)
    print('= ' * 20)