import numpy as np
import threading

class PheromoneMap:
    def __init__(self, width, height, init=0.0):
        self.width = width
        self.height = height
        # Dual pheromone layers: exploration trails and victim-path trails
        self.exploration_grid = np.full((height, width), float(init), dtype=float)
        self.victim_path_grid = np.full((height, width), float(init), dtype=float)
        # Repulsive pheromones for dead-ends
        self.repulsive_grid = np.full((height, width), float(init), dtype=float)
        # Visit count tracking for adaptive evaporation
        self.visit_counts = np.zeros((height, width), dtype=int)
        self.lock = threading.Lock()

    def get(self, x, y):
        """Get exploration pheromone strength"""
        return float(self.exploration_grid[y, x])
    
    def get_victim_path(self, x, y):
        """Get victim-path pheromone strength (2x importance)"""
        return float(self.victim_path_grid[y, x])
    
    def get_repulsive(self, x, y):
        """Get repulsive pheromone (negative influence)"""
        return float(self.repulsive_grid[y, x])
    
    def get_combined(self, x, y):
        """Get combined pheromone considering all types"""
        exploration = self.exploration_grid[y, x]
        victim_path = self.victim_path_grid[y, x] * 2.0  # 2x strength
        repulsive = -self.repulsive_grid[y, x]  # negative influence
        return float(exploration + victim_path + repulsive)

    def deposit(self, x, y, Q, pheromone_type='exploration'):
        """Deposit pheromone of specific type"""
        with self.lock:
            if pheromone_type == 'exploration':
                self.exploration_grid[y, x] += Q
            elif pheromone_type == 'victim_path':
                self.victim_path_grid[y, x] += Q
            elif pheromone_type == 'repulsive':
                self.repulsive_grid[y, x] += Q
            self.visit_counts[y, x] += 1

    def evaporate(self, rho, adaptive=True):
        """Evaporate pheromones with optional adaptive rates"""
        with self.lock:
            if adaptive:
                # Adaptive evaporation: faster decay in well-covered areas, slower in sparse regions
                # Well-covered: visit_count > 3 -> rho*2 (faster decay)
                # Sparse: visit_count <= 1 -> rho*0.5 (slower decay)
                rho_map = np.where(self.visit_counts > 3, rho * 2.0,
                          np.where(self.visit_counts <= 1, rho * 0.5, rho))
                self.exploration_grid *= (1.0 - rho_map)
                self.victim_path_grid *= (1.0 - rho_map * 0.5)  # victim paths decay slower
                self.repulsive_grid *= (1.0 - rho_map * 1.5)  # repulsive decays faster
            else:
                self.exploration_grid *= (1.0 - rho)
                self.victim_path_grid *= (1.0 - rho * 0.5)
                self.repulsive_grid *= (1.0 - rho * 1.5)

    def merge(self, other_exploration, other_victim_path=None, other_repulsive=None):
        """Merge pheromone maps from other robots"""
        with self.lock:
            self.exploration_grid = np.maximum(self.exploration_grid, other_exploration)
            if other_victim_path is not None:
                self.victim_path_grid = np.maximum(self.victim_path_grid, other_victim_path)
            if other_repulsive is not None:
                self.repulsive_grid = np.maximum(self.repulsive_grid, other_repulsive)

    def copy(self):
        """Return combined pheromone grid for visualization"""
        with self.lock:
            return self.exploration_grid.copy() + self.victim_path_grid.copy() * 2.0
    
    def copy_all(self):
        """Return all pheromone layers"""
        with self.lock:
            return {
                'exploration': self.exploration_grid.copy(),
                'victim_path': self.victim_path_grid.copy(),
                'repulsive': self.repulsive_grid.copy(),
                'visit_counts': self.visit_counts.copy()
            }
