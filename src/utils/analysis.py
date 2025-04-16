import os
import numpy as np
from collections import deque

def load_binary_grid(path):
    with open(path, 'r') as f:
        lines = f.read().strip().split('\n')
        grid = []
        for line in lines:
            tokens = line.strip().replace('O', '0').split()
            if len(tokens) == 1 and all(c in '01' for c in tokens[0]):
                tokens = list(tokens[0])
            row = [int(t) if t in ('0', '1') else 0 for t in tokens]
            grid.append(row)
        return np.array(grid)

def compute_density(grid):
    total = grid.size
    wall_count = np.count_nonzero(grid == 1)
    return wall_count / total if total > 0 else 0

def count_connected_wall_components(grid, connectivity=8):
    visited = np.zeros_like(grid, dtype=bool)
    rows, cols = grid.shape
    count = 0

    if connectivity == 8:
        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
    else:
        directions = [(-1,0), (1,0), (0,-1), (0,1)]

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1 and not visited[i][j]:
                count += 1
                q = deque([(i, j)])
                visited[i][j] = True
                while q:
                    x, y = q.popleft()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols:
                            if grid[nx][ny] == 1 and not visited[nx][ny]:
                                visited[nx][ny] = True
                                q.append((nx, ny))
    return count

def shortest_path_length(grid):
    if grid[0][0] == 1 or grid[-1][-1] == 1:
        return None

    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    q = deque([(0, 0, 0)])
    directions = [(-1,0), (1,0), (0,-1), (0,1)]

    while q:
        x, y, d = q.popleft()
        if x == rows - 1 and y == cols - 1:
            return d
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < rows) and (0 <= ny < cols):
                if not visited[nx][ny] and grid[nx][ny] == 0:
                    visited[nx][ny] = True
                    q.append((nx, ny, d + 1))
    return None

def analyze_directory(base_dir, connectivity=8):
    densities = []
    wall_components = []
    path_lengths = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if not file.endswith('.txt'):
                continue
            path = os.path.join(root, file)
            try:
                grid = load_binary_grid(path)
                densities.append(compute_density(grid))
                wall_components.append(count_connected_wall_components(grid, connectivity))
                path_len = shortest_path_length(grid)
                if path_len is not None:
                    path_lengths.append(path_len)
            except Exception as e:
                print(f"Skipping {path}: {e}")

    avg_density = sum(densities) / len(densities) if densities else 0
    avg_components = sum(wall_components) / len(wall_components) if wall_components else 0
    avg_path = sum(path_lengths) / len(path_lengths) if path_lengths else 0

    return {
        "avg_density": avg_density,
        "avg_wall_components": avg_components,
        "avg_shortest_path_length": avg_path,
        "grids_with_valid_path": len(path_lengths)
    }

# ✅ Example usage
for i in [0,25,50,75,95]:
    try:
        env_dir = f"../assets/with_ga/racing_track_results_original/ga/generation{i}/samples/txt/"
        result = analyze_directory(env_dir, connectivity=8)

        print(f"Analysis Summary generation {i}:")   
        print(f"Average Wall Density: {result['avg_density']:.2%}")
        print(f"Avg # of Connected Wall Components: {result['avg_wall_components']:.2f}")
        print(f"Average Shortest Path (if exists): {result['avg_shortest_path_length']:.2f} steps")
        print(f"Grids with Valid Path: {result['grids_with_valid_path']}")
    except Exception as e:
        pass
