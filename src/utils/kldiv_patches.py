# Re-executing after kernel reset

import os
import numpy as np
from collections import Counter

def load_grid_from_txt(path, grid_size=128):
    with open(path, 'r') as f:
        lines = [line.strip().replace('O', '0') for line in f if line.strip()]
    grid = []
    for line in lines:
        tokens = line.split()
        if len(tokens) == 1 and len(tokens[0]) == grid_size:
            tokens = list(tokens[0])
        if len(tokens) == grid_size:
            row = [int(t) if t in ('0', '1') else 0 for t in tokens]
            grid.append(row)
    grid = np.array(grid)
    if grid.shape != (grid_size, grid_size):
        raise ValueError(f"Invalid shape: {grid.shape}")
    return grid

def load_all_grids_from_dir(directory, grid_size=128):
    grids = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".txt"):
                try:
                    grid = load_grid_from_txt(os.path.join(root, filename), grid_size)
                    grids.append(grid)
                except:
                    continue
    return grids

def extract_nonoverlapping_patch_patterns(grid, patch_size):
    patterns = []
    rows, cols = grid.shape
    for i in range(0, rows - patch_size + 1, patch_size):
        for j in range(0, cols - patch_size + 1, patch_size):
            patch = grid[i:i+patch_size, j:j+patch_size]
            patterns.append(tuple(patch.flatten()))
    return patterns

def compute_pattern_distribution(grids, patch_size):
    counter = Counter()
    for grid in grids:
        patterns = extract_nonoverlapping_patch_patterns(grid, patch_size)
        counter.update(patterns)
    total = sum(counter.values())
    return {k: v / total for k, v in counter.items()}

def compute_kl_divergence(P, Q, epsilon=1e-8):
    all_keys = set(P.keys()) | set(Q.keys())
    return sum(P.get(k, epsilon) * np.log(P.get(k, epsilon) / Q.get(k, epsilon)) for k in all_keys)

def compute_average_patch_kldiv(ref_grids, gen_grids, patch_sizes=[2, 3, 4]):
    return np.mean([
        compute_kl_divergence(
            compute_pattern_distribution(ref_grids, size),
            compute_pattern_distribution(gen_grids, size)
        )
        for size in patch_sizes
    ])
# Example usage
for i in range(1,4):
    if i ==1 :
        ref_dir = "../assets/data/racing_track"
       
    else:
        ref_dir = f"../assets/data/racing_track{i}"
        
    ref_grids = load_all_grids_from_dir(ref_dir)
    for j in range(1,4):
        if j ==1:
            gen_dir = "../assets/racing_track_results/samples/txt"
        else:
            gen_dir = f"../assets/racing_track{j}_results/samples/txt"

        
        gen_grids = load_all_grids_from_dir(gen_dir)

        if ref_grids and gen_grids:
            patch_sizes=[2,3,4]
            avg_kldiv = compute_average_patch_kldiv(ref_grids, gen_grids, patch_sizes=patch_sizes)
            print(f"Mean Patch-Based KL-Divergence (patch_sizes) between ref {i} and gen {j}:", avg_kldiv)
        else:
            print("No valid grids found in one or both directories.")
