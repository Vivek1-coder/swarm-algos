import threading
import random
import math
import time
import numpy as np

class Robot(threading.Thread):
    def __init__(self, rid, env, pheromap, topics, start_pos=(0,0),
                 comm_range=8, alpha=0.5, beta=3.0, rho=0.05, Q=1.0, energy=500,
                 sector=None, total_steps=3000):
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
        # Base ACO parameters (will be adapted dynamically)
        self.base_alpha = alpha
        self.base_beta = beta
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.energy = energy
        self.max_energy = energy
        self.energy_threshold = 50  # recharge when energy below this
        # Local pheromone maps (all layers)
        self.local_exploration = np.zeros((env.height, env.width), dtype=float)
        self.local_victim_path = np.zeros((env.height, env.width), dtype=float)
        self.local_repulsive = np.zeros((env.height, env.width), dtype=float)
        self.detected_victims = set()
        self.failed = False
        self.running = True
        # Sector assignment for coordinated dispersion
        self.sector = sector  # tuple (min_x, max_x, min_y, max_y)
        # Simulation tracking
        self.current_step = 0
        self.total_steps = total_steps
        self.unexplored_bonus = 4.0  # will be adapted
        # Robot density tracking for spreading pressure
        self.nearby_robots = []
        # Victim trail tracking
        self.on_victim_trail = False
        self.victim_trail_path = []
        # Exploration state tracking
        self.fully_explored_areas = set()  # cells marked as fully explored by other robots

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

    def deposit_pheromone(self, pheromone_type='exploration'):
        """Deposit pheromone with type differentiation"""
        self.pheromap.deposit(self.x, self.y, self.Q, pheromone_type=pheromone_type)
        
        # Update local map
        if pheromone_type == 'exploration':
            self.local_exploration[self.y, self.x] = max(self.local_exploration[self.y, self.x], 
                                                          self.pheromap.get(self.x, self.y))
        elif pheromone_type == 'victim_path':
            self.local_victim_path[self.y, self.x] = max(self.local_victim_path[self.y, self.x],
                                                          self.pheromap.get_victim_path(self.x, self.y))
        elif pheromone_type == 'repulsive':
            self.local_repulsive[self.y, self.x] = max(self.local_repulsive[self.y, self.x],
                                                        self.pheromap.get_repulsive(self.x, self.y))
        
        self.visited.add((self.x, self.y))
        
        # Mark neighbors as partially explored (reduce revisit bias)
        for nx, ny in self.env.neighbors(self.x, self.y, radius=1):
            if not self.env.is_obstacle(nx, ny):
                self.local_exploration[ny, nx] = max(self.local_exploration[ny, nx], 0.01)

    def publish_local_map(self):
        """Publish all local pheromone layers to map_merge topic"""
        if "map_merge" in self.topics:
            self.topics["map_merge"].publish({
                "robot": self.rid,
                "exploration": self.local_exploration.copy(),
                "victim_path": self.local_victim_path.copy(),
                "repulsive": self.local_repulsive.copy()
            })

    def update_phase_parameters(self):
        """Dynamically adjust ACO parameters based on simulation phase"""
        phase_progress = self.current_step / max(1, self.total_steps)
        
        if phase_progress < 0.30:
            # Early phase: HIGH EXPLORATION
            self.alpha = 0.3  # reduce pheromone influence
            self.unexplored_bonus = 6.0  # increase unexplored bonus
        elif phase_progress < 0.70:
            # Mid phase: BALANCED
            self.alpha = self.base_alpha  # 0.5
            self.unexplored_bonus = 4.0
        else:
            # Late phase: HIGH EXPLOITATION
            self.alpha = 0.8  # increase pheromone influence
            self.unexplored_bonus = 2.0  # reduce unexplored bonus

    def get_robot_density(self, x, y, radius=3):
        """Calculate nearby robot density for spreading pressure"""
        count = 0
        for other in self.nearby_robots:
            dist = abs(other[0] - x) + abs(other[1] - y)  # Manhattan distance
            if dist <= radius:
                count += 1
        return count

    def choose_next_cell(self):
        """Enhanced ACO with adaptive parameters, victim trails, and spreading pressure"""
        self.update_phase_parameters()
        
        # Get candidates within comm_range, but prioritize closer cells for efficiency
        neighbors = []
        for radius in [1, 2, 3]:  # prioritize immediate neighbors
            for nx, ny in self.env.neighbors(self.x, self.y, radius=radius):
                if not self.env.is_obstacle(nx, ny):
                    neighbors.append((nx, ny))
            if len(neighbors) >= 8:  # sufficient candidates
                break
        
        # Extend to comm_range if needed
        if len(neighbors) < 8:
            for nx in range(max(0, self.x - self.comm_range), min(self.env.width, self.x + self.comm_range + 1)):
                for ny in range(max(0, self.y - self.comm_range), min(self.env.height, self.y + self.comm_range + 1)):
                    if nx == self.x and ny == self.y:
                        continue
                    if self.env.is_obstacle(nx, ny):
                        continue
                    if (nx, ny) not in neighbors:
                        neighbors.append((nx, ny))
        
        if not neighbors:
            return (self.x, self.y)
        
        # Compute pheromone values (exploration + victim_path + repulsive)
        tau_vals = np.array([self.pheromap.get_combined(nx, ny) for (nx, ny) in neighbors], dtype=float)
        
        # Enhanced heuristic with multiple factors
        eta_vals = []
        for nx, ny in neighbors:
            # Base: distance encourages spreading
            heuristic = 1.0 + 0.5 * (abs(nx - self.x) + abs(ny - self.y))
            
            # STRONG unexplored bonus (adaptive based on phase)
            if self.local_exploration[ny, nx] == 0 and (nx, ny) not in self.fully_explored_areas:
                heuristic += self.unexplored_bonus
            
            # Victim path following bonus (2x weight when on trail)
            victim_pher = self.pheromap.get_victim_path(nx, ny)
            if victim_pher > 0.1:
                heuristic += self.beta * victim_pher  # high priority for victim trails
                self.on_victim_trail = True
            
            # Sector preference (guide toward assigned sector)
            if self.sector is not None:
                min_x, max_x, min_y, max_y = self.sector
                if min_x <= nx < max_x and min_y <= ny < max_y:
                    heuristic += 2.0  # bonus for staying in sector
            
            # Spreading pressure: penalize high robot density areas
            density = self.get_robot_density(nx, ny, radius=3)
            heuristic -= density * 1.5  # reduce appeal of crowded areas
            
            # Collision avoidance: strong penalty for 2-cell radius
            density_close = self.get_robot_density(nx, ny, radius=2)
            if density_close > 0:
                heuristic -= density_close * 3.0
            
            # Repulsive pheromone penalty (dead-ends)
            repulsive = self.pheromap.get_repulsive(nx, ny)
            heuristic -= repulsive * 2.0
            
            # Energy-aware: when low energy, favor paths toward base
            if self.energy < 100:
                dist_to_base = abs(nx) + abs(ny)
                current_dist = abs(self.x) + abs(self.y)
                if dist_to_base < current_dist:
                    heuristic += 3.0  # strong bonus for moving toward base
            
            eta_vals.append(max(0.1, heuristic))  # ensure positive
        
        eta_vals = np.array(eta_vals, dtype=float)
        
        # ACO formula: (tau^alpha) * (eta^beta)
        numerators = np.power(tau_vals + 0.01, self.alpha) * np.power(eta_vals, self.beta)
        
        if numerators.sum() == 0:
            probs = np.ones(len(neighbors)) / len(neighbors)
        else:
            probs = numerators / numerators.sum()
        
        choice = random.choices(neighbors, weights=probs, k=1)[0]
        return choice

    def sense_victim_extended(self):
        """Extended victim search with increased radius"""
        max_conf = 0.0
        victim_pos = None
        
        # Check current cell (highest confidence)
        if self.env.has_victim(self.x, self.y):
            return random.uniform(0.8, 1.0), (self.x, self.y)
        
        # Check extended radius (reduced confidence with distance)
        for radius in [1, 2, 3]:
            for nx, ny in self.env.neighbors(self.x, self.y, radius=radius):
                if self.env.has_victim(nx, ny):
                    conf = random.uniform(0.4, 0.7) / radius  # confidence decreases with distance
                    if conf > max_conf:
                        max_conf = conf
                        victim_pos = (nx, ny)
        
        return max_conf, victim_pos

    def deposit_victim_trail(self, victim_x, victim_y):
        """Deposit strong directional pheromone trail back to base when victim found"""
        # Create trail from victim to current position
        path = [(victim_x, victim_y)]
        cx, cy = victim_x, victim_y
        
        # Simple pathfinding toward current position
        while (cx, cy) != (self.x, self.y) and len(path) < 50:
            if cx < self.x and not self.env.is_obstacle(cx + 1, cy):
                cx += 1
            elif cx > self.x and not self.env.is_obstacle(cx - 1, cy):
                cx -= 1
            elif cy < self.y and not self.env.is_obstacle(cx, cy + 1):
                cy += 1
            elif cy > self.y and not self.env.is_obstacle(cx, cy - 1):
                cy -= 1
            else:
                break
            path.append((cx, cy))
        
        # Deposit strong victim-path pheromones along trail
        for px, py in path:
            self.pheromap.deposit(px, py, self.Q * 3.0, pheromone_type='victim_path')
            self.local_victim_path[py, px] = self.pheromap.get_victim_path(px, py)

    def detect_dead_end(self):
        """Detect if current position is a dead-end (deposit repulsive pheromone)"""
        free_neighbors = [n for n in self.env.neighbors(self.x, self.y, radius=1) 
                         if not self.env.is_obstacle(n[0], n[1])]
        
        if len(free_neighbors) <= 1:  # dead-end or corridor end
            self.pheromap.deposit(self.x, self.y, 0.5, pheromone_type='repulsive')
            self.local_repulsive[self.y, self.x] = self.pheromap.get_repulsive(self.x, self.y)
            return True
        return False

    def predictive_recharge_needed(self):
        """Check if robot should return for recharge based on distance to base"""
        dist_to_base = abs(self.x) + abs(self.y)  # Manhattan distance
        energy_needed = dist_to_base * 1.5  # safety margin
        return self.energy < energy_needed

    def run(self):
        """Enhanced main robot loop with all improvements"""
        while self.running and not self.failed:
            self.current_step += 1
            
            # Update nearby robots list from communications (robot positions)
            if "robot_positions" in self.topics:
                # This would be populated by a position broadcast system
                pass
            
            # Simulate sensor sensing
            if self.sense_obstacle():
                self.avoidance_behavior()
                self.publish_local_map()
                time.sleep(0.01)
                continue

            # Extended victim detection
            conf, victim_pos = self.sense_victim_extended()
            if conf > 0.5:  # significant detection
                victim_to_report = None
                if victim_pos:
                    vx, vy = victim_pos
                    victim_to_report = (vx, vy)
                else:
                    victim_to_report = (self.x, self.y)
                
                # Only report if not already detected
                if victim_to_report not in self.detected_victims:
                    self.detected_victims.add(victim_to_report)
                    self.broadcast_beacon(conf)
                    # Deposit strong victim-path trail
                    if victim_pos:
                        self.deposit_victim_trail(vx, vy)
                    self.deposit_pheromone(pheromone_type='victim_path')
                    self.on_victim_trail = True
                    self.publish_local_map()
                
                # Don't get stuck - continue moving after detection
                time.sleep(0.01)

            # Detect dead-ends and mark with repulsive pheromone
            self.detect_dead_end()

            # Predictive energy management
            if self.predictive_recharge_needed():
                # Navigate toward base efficiently
                if self.x > 0 or self.y > 0:
                    # Increase alpha to favor known paths
                    old_alpha = self.alpha
                    self.alpha = min(0.9, self.alpha + 0.3)
                    
                    # Choose path toward base
                    candidates = []
                    for nx, ny in self.env.neighbors(self.x, self.y, radius=1):
                        if not self.env.is_obstacle(nx, ny):
                            dist = abs(nx) + abs(ny)
                            if dist < (abs(self.x) + abs(self.y)):
                                candidates.append((nx, ny))
                    
                    if candidates:
                        self.x, self.y = random.choice(candidates)
                    else:
                        self.x, self.y = self.choose_next_cell()
                    
                    self.alpha = old_alpha
                    self.energy -= 1
                    self.deposit_pheromone()
                    self.publish_local_map()
                    time.sleep(0.01)
                    continue
                else:
                    # At base, recharge
                    self.energy = self.max_energy
                    time.sleep(0.01)
                    continue

            # ACO-based exploration with reduced random jitter (2%)
            if random.random() < 0.02:
                neighbors = [n for n in self.env.neighbors(self.x, self.y, radius=1) 
                           if not self.env.is_obstacle(n[0], n[1])]
                if neighbors:
                    nx, ny = random.choice(neighbors)
                else:
                    nx, ny = self.choose_next_cell()
            else:
                nx, ny = self.choose_next_cell()
            
            # Move
            self.x, self.y = nx, ny
            self.energy -= 1
            
            # Deposit appropriate pheromone type
            if self.on_victim_trail:
                self.deposit_pheromone(pheromone_type='victim_path')
                self.on_victim_trail = False  # reset after one step
            else:
                self.deposit_pheromone(pheromone_type='exploration')
            
            # Publish local map periodically
            if self.current_step % 10 == 0:  # reduce communication overhead
                self.publish_local_map()

            # Virtual pheromone sync every 100 steps
            if self.current_step % 100 == 0:
                self.publish_local_map()  # full sync
                # Broadcast fully explored areas
                explored_threshold = 5
                for (ex, ey), count in zip(self.visited, [1] * len(self.visited)):
                    if self.local_exploration[ey, ex] > explored_threshold:
                        if "area_explored" in self.topics:
                            self.topics["area_explored"].publish({
                                "robot": self.rid,
                                "cell": (ex, ey),
                                "confidence": 1.0
                            })

            # Random failure simulation (very low rate)
            if random.random() < 0.00001:
                self.failed = True
                if "status" in self.topics:
                    self.topics["status"].publish({
                        "robot": self.rid,
                        "failed": True,
                        "pos": (self.x, self.y)
                    })
                break

            time.sleep(0.01)

    def stop(self):
        self.running = False

