import os
import numpy as np
from collections import defaultdict

def load_grid_from_txt(path, grid_size=32):
    grid = []
    with open(path, 'r') as f:
        lines = f.read().strip().split('\n')
        for line_num, line in enumerate(lines):
            line = line.strip().replace('O', '0')  # Handle common typo
            tokens = line.split()

            # If line is a single long string, break it into characters
            if len(tokens) == 1 and len(tokens[0]) == grid_size:
                tokens = list(tokens[0])
            elif len(tokens) != grid_size:
                print(f"Line {line_num + 1} has {len(tokens)} tokens, expected {grid_size}. Skipping.")
                continue

            try:
                row = [int(token) if token in ('0', '1') else 0 for token in tokens]
                if len(row) == grid_size:
                    grid.append(row)
            except ValueError as e:
                print(f"Error parsing line {line_num + 1}: {e}")
                continue

    grid = np.array(grid)
    if grid.shape != (grid_size, grid_size):
        raise ValueError(f"❌ Grid in {path} has invalid shape: {grid.shape}")
    return grid



def load_all_grids_from_dir(directory, grid_size=32):
    grids = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".txt"):
                path = os.path.join(root, filename)
                try:
                    grid = load_grid_from_txt(path)
                    if grid.shape == (grid_size, grid_size):
                        grids.append(grid)
                    else:
                        print(f"Skipping {path}: Unexpected shape {grid.shape}")
                except Exception as e:
                    print(f"Skipping {path}: {e}")
    return grids

def compute_cooccurrence(grid, directions=[(0, 1), (1, 0), (1, 1), (1, -1)]):
    counts = defaultdict(int)
    for dx, dy in directions:
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                    a, b = grid[x][y], grid[nx][ny]
                    counts[(a, b, dx, dy)] += 1

    probs = {}
    for dx, dy in directions:
        dir_total = sum(counts[(a, b, dx, dy)] for a in [0, 1] for b in [0, 1])
        for a in [0, 1]:
            for b in [0, 1]:
                key = (a, b, dx, dy)
                probs[key] = counts[key] / dir_total if dir_total > 0 else 0
    return probs

def compute_tpkldiv(ref_grid, gen_grids):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    epsilon = 1e-8

    ref_probs = compute_cooccurrence(ref_grid, directions)
    avg_gen_probs = defaultdict(float)

    for grid in gen_grids:
        gen_probs = compute_cooccurrence(grid, directions)
        for key in gen_probs:
            avg_gen_probs[key] += gen_probs[key]
    for key in avg_gen_probs:
        avg_gen_probs[key] /= len(gen_grids)

    kls = []
    for dx, dy in directions:
        kl = 0.0
        for a in [0, 1]:
            for b in [0, 1]:
                key = (a, b, dx, dy)
                p = max(ref_probs.get(key, 0), epsilon)
                q = max(avg_gen_probs.get(key, 0), epsilon)
                kl += p * np.log(p / q)
        kls.append(kl)

    return sum(kls) / len(kls)

# Usage
original_path = '../assets/data/train4/factory.txt'
generated_dir = '../assets/results16/ga/generation37/agent0/txt'  # directory containing .txt files

ref_grid = load_grid_from_txt(original_path)
gen_grids = load_all_grids_from_dir(generated_dir)

if gen_grids:
    tpkldiv = compute_tpkldiv(ref_grid, gen_grids)
    print("TPKL-Div:", tpkldiv)
else:
    print("No valid generated grids found.")
