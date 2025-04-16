import os
from collections import defaultdict
import numpy as np

def load_token_grid_from_txt(path):
    with open(path, 'r') as f:
        lines = f.read().strip().split('\n')
        grid = [line.strip().split() for line in lines if line.strip()]
    return grid

def extract_non_overlapping_patches(grid, patch_size):
    rows, cols = len(grid), len(grid[0])
    if rows < patch_size or cols < patch_size:
        return []  # Grid too small for this patch size

    patches = []
    for i in range(0, rows - patch_size + 1, patch_size):
        for j in range(0, cols - patch_size + 1, patch_size):
            patch = tuple(tuple(grid[i + x][j + y] for y in range(patch_size)) for x in range(patch_size))
            patches.append(patch)
    return patches

def compute_uniqueness_non_overlapping(dir_path, patch_sizes=[3, 5, 7]):
    patch_sets = {size: set() for size in patch_sizes}
    patch_totals = {size: 0 for size in patch_sizes}

    for root, _, files in os.walk(dir_path):
        for filename in files:
            if not filename.endswith('.txt'):
                continue
            try:
                grid = load_token_grid_from_txt(os.path.join(root, filename))
                for size in patch_sizes:
                    patches = extract_non_overlapping_patches(grid, size)
                    patch_sets[size].update(patches)
                    patch_totals[size] += len(patches)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    uniqueness_per_size = {
        size: (len(patch_sets[size]) / patch_totals[size]) if patch_totals[size] > 0 else 0
        for size in patch_sizes
    }
    avg_uniqueness = (
        sum(uniqueness_per_size.values()) / len(uniqueness_per_size)
        if uniqueness_per_size else 0
    )

    return uniqueness_per_size, avg_uniqueness
# Example usage:
env_dir = "../assets/factory1_results/ga/generation95/samples/txt"
env_dir = "../assets/dron1_results/random_samples/txt"
sizes = [6, 8]
uniq_by_size, avg = compute_uniqueness_non_overlapping(env_dir, sizes)

print("Tile Uniqueness by Patch Size:")
for sz, val in uniq_by_size.items():
    print(f"  {sz}x{sz}: {val:.2%}")
print(f"Average Uniqueness: {avg:.2%}")
