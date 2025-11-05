import threading
import random
import math
import time
import numpy as np

class Robot(threading.Thread):
    def __init__(self, rid, env, pheromap, topics, start_pos=(0,0),
                 comm_range=8, alpha=0.5, beta=3.0, rho=0.05, Q=1.0, energy=500):
        super().__init__(daemon=True)
        self.rid = rid
        self.env = env
        self.pheromap = pheromap
        self.topics = topics  # dict of Topic instances
        self.x, self.y = start_pos
        # visited cells set for coverage computation
        self.visited = set()
        self.visited.add((self.x, self.y))
        self.comm_range = comm_range
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.energy = energy
        self.energy_threshold = 50  # recharge when energy below this
        self.local_map = np.zeros((env.height, env.width), dtype=float)
        self.detected_victims = set()
        self.failed = False
        self.running = True
        # subscribe to global pheromone broadcasts (optional)
        # subscribe to map merge requests (we publish our local map to "map_merge")

    # Simulated sensors
    def sense_obstacle(self):
        # LIDAR: check immediate cell ahead and neighbors
        return self.env.is_obstacle(self.x, self.y)

    def sense_victim(self):
        # Thermal/sound: probabilistic detection if victim in same cell
        if self.env.has_victim(self.x, self.y):
            # return confidence [0,1]
            return random.uniform(0.6, 1.0)
        # possible remote detections with low confidence
        return 0.0

    def avoidance_behavior(self):
        # simple random sidestep to any non-obstacle neighbor
        for nx, ny in random.sample(self.env.neighbors(self.x, self.y, radius=1),
                                     k=len(self.env.neighbors(self.x, self.y, radius=1))):
            if not self.env.is_obstacle(nx, ny):
                self.x, self.y = nx, ny
                # mark visited
                self.visited.add((self.x, self.y))
                self.energy -= 1
                return
        # if stuck, stay

    def broadcast_beacon(self, conf ):
        # publish a beacon with detection confidence; subscribers may use it
        msg = {"robot": self.rid, "pos": (self.x, self.y), "conf": float(conf)}
        if "beacon" in self.topics:
            self.topics["beacon"].publish(msg)

    def deposit_pheromone(self):
        self.pheromap.deposit(self.x, self.y, self.Q)
        # mark local map pheromone and visited
        self.local_map[self.y, self.x] = max(self.local_map[self.y, self.x], self.pheromap.get(self.x, self.y))
        self.visited.add((self.x, self.y))
        # also mark neighbors as explored in local map (to reduce revisit bias)
        for nx, ny in self.env.neighbors(self.x, self.y, radius=1):
            if not self.env.is_obstacle(nx, ny):
                self.local_map[ny, nx] = max(self.local_map[ny, nx], 0.01)

    def publish_local_map(self):
        # publish local map to map_merge topic
        if "map_merge" in self.topics:
            self.topics["map_merge"].publish({"robot": self.rid, "map": self.local_map.copy()})

    def choose_next_cell(self):
        # ACO: compute neighboring cells within comm_range
        neighbors = []
        for nx in range(max(0, self.x - self.comm_range), min(self.env.width, self.x + self.comm_range + 1)):
            for ny in range(max(0, self.y - self.comm_range), min(self.env.height, self.y + self.comm_range + 1)):
                if nx == self.x and ny == self.y:
                    continue
                if self.env.is_obstacle(nx, ny):
                    continue
                neighbors.append((nx, ny))
        if not neighbors:
            return (self.x, self.y)
        
        tau_vals = np.array([self.pheromap.get(nx, ny) for (nx, ny) in neighbors], dtype=float)
        
        # Enhanced heuristic for balanced explore-exploit:
        # - Unexplored cells (local_map[ny,nx]==0) get strong priority
        # - Distance bonus encourages spreading away from base
        # - Pheromone bonus rewards trail following (exploitation)
        # - Combination creates natural explore-first, then exploit trails
        
        eta_vals = np.array([
            (1.0 + 0.5 * (abs(nx - self.x) + abs(ny - self.y)))  # distance favor: prefer farther cells
            + (4.0 if self.local_map[ny, nx] == 0 else 0.0)      # STRONG unexplored bonus
            + (0.5 * (self.pheromap.get(nx, ny) + 0.01))         # small pheromone bonus for exploitation
            for (nx, ny) in neighbors], dtype=float)
        
        # ACO formula: (tau^alpha) * (eta^beta)
        # alpha=0.5 (low pheromone weight allows exploration early)
        # beta=3.0 (high heuristic weight prioritizes unexplored+distance)
        numerators = np.power(tau_vals + 0.01, self.alpha) * np.power(eta_vals, self.beta)
        
        if numerators.sum() == 0:
            probs = np.ones(len(neighbors)) / len(neighbors)
        else:
            probs = numerators / numerators.sum()
        
        choice = random.choices(neighbors, weights=probs, k=1)[0]
        return choice

    def run(self):
        # main robot loop
        step = 0
        while self.running and not self.failed:
            step += 1
            # simulate sensor sensing
            if self.sense_obstacle():
                self.avoidance_behavior()
                self.publish_local_map()
                time.sleep(0.01)
                continue

            conf = self.sense_victim()
            if conf > 0.0:
                # victim detected
                self.detected_victims.add((self.x, self.y))
                self.broadcast_beacon(conf)
                self.deposit_pheromone()
                # search locally for confirmation by checking neighbors
                for nx, ny in self.env.neighbors(self.x, self.y, radius=1):
                    if self.env.has_victim(nx, ny):
                        self.detected_victims.add((nx, ny))
                        self.broadcast_beacon(random.uniform(0.6, 1.0))
                self.publish_local_map()
                time.sleep(0.01)
                continue

            # ACO-based exploration
            # with small probability, take random action to escape local optima
            if random.random() < 0.02:  # 2% chance of random move (reduced from 5%)
                neighbors = [n for n in self.env.neighbors(self.x, self.y, radius=1) if not self.env.is_obstacle(n[0], n[1])]
                if neighbors:
                    nx, ny = random.choice(neighbors)
                else:
                    nx, ny = self.choose_next_cell()
            else:
                nx, ny = self.choose_next_cell()
            
            # move
            self.x, self.y = nx, ny
            self.energy -= 1
            # deposit pheromone at arrival
            self.deposit_pheromone()
            # publish local map to merge
            self.publish_local_map()

            # random simulation of communication loss or failure
            if random.random() < 0.00001:  # very low failure rate to avoid early termination
                # small chance to fail
                self.failed = True
                if "status" in self.topics:
                    self.topics["status"].publish({"robot": self.rid, "failed": True, "pos": (self.x, self.y)})
                break

            # energy check
            if self.energy < self.energy_threshold:
                # return to base (0,0) to recharge
                # navigate toward base
                if self.x > 0 or self.y > 0:
                    # move closer to base
                    if self.x > 0:
                        self.x -= 1
                    elif self.y > 0:
                        self.y -= 1
                    self.energy -= 1
                else:
                    # at base, recharge
                    self.energy = 500
                # deposit some pheromone at base
                self.deposit_pheromone()
                self.publish_local_map()

            time.sleep(0.01)

    def stop(self):
        self.running = False

