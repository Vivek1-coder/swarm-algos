import numpy as np
import threading

class PheromoneMap:
    def __init__(self, width, height, init=0.0):
        self.width = width
        self.height = height
        self.grid = np.full((height, width), float(init), dtype=float)
        self.lock = threading.Lock()

    def get(self, x, y):
        return float(self.grid[y, x])

    def deposit(self, x, y, Q):
        with self.lock:
            self.grid[y, x] += Q

    def evaporate(self, rho):
        with self.lock:
            self.grid *= (1.0 - rho)

    def merge(self, other_grid):
        # merge by taking max pheromone per cell
        with self.lock:
            self.grid = np.maximum(self.grid, other_grid)

    def copy(self):
        with self.lock:
            return self.grid.copy()
