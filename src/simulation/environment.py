import numpy as np
import random

class Environment:
    def __init__(self, width=50, height=30, obstacle_prob=0.08, victim_count=5, seed=None):
        self.width = width
        self.height = height
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.obstacles = np.zeros((height, width), dtype=bool)
        self.victims = np.zeros((height, width), dtype=bool)
        # populate obstacles
        self.obstacles = np.random.rand(height, width) < obstacle_prob
        # ensure base at (0,0) is free
        self.obstacles[0,0] = False
        # place victims randomly in free cells
        free_cells = list(zip(*np.where(~self.obstacles)))
        random.shuffle(free_cells)
        for i in range(min(victim_count, len(free_cells))):
            y,x = free_cells[i]
            self.victims[y,x] = True

    def is_obstacle(self, x, y):
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return True
        return bool(self.obstacles[y,x])

    def has_victim(self, x, y):
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return False
        return bool(self.victims[y,x])

    def remove_victim(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.victims[y,x] = False

    def neighbors(self, x, y, radius=1):
        coords = []
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                nx, ny = x+dx, y+dy
                if dx==0 and dy==0:
                    continue
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    coords.append((nx, ny))
        return coords
