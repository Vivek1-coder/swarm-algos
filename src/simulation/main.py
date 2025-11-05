import argparse
import time
import threading
import math
import random
from .environment import Environment
from .pheromone_map import PheromoneMap
from .communications import Topic
from .robot import Robot
from .visualize import Visualizer


def run_simulation(steps=200, robots_count=4, width=60, height=36):
    env = Environment(width=width, height=height, obstacle_prob=0.07, victim_count=6)
    pheromap = PheromoneMap(width, height, init=0.0)
    # topics
    topics = {
        "beacon": Topic(),
        "map_merge": Topic(),
        "status": Topic(),
        "robot_positions": Topic(),
        "area_explored": Topic()
    }

    # confirmed victims set (shared with visualizer)
    confirmed_victims = set()
    
    # Track time to first victim discovery
    first_victim_time = None
    start_time = time.time()

    # central map merge handler - now handles all pheromone layers
    def handle_map(msg):
        # msg: {'robot': id, 'exploration': np.array, 'victim_path': np.array, 'repulsive': np.array}
        if 'exploration' in msg:
            pheromap.merge(msg['exploration'], 
                          msg.get('victim_path'), 
                          msg.get('repulsive'))

    topics['map_merge'].subscribe(handle_map)

    # beacon subscriber: log detection events and record beacon messages
    beacon_events = []
    import time as _time
    def beacon_handler(msg):
        nonlocal first_victim_time
        # msg: {"robot": id, "pos": (x,y), "conf": float}
        t = _time.time()
        beacon_events.append({"time": t, "robot": msg.get('robot'), "pos": msg.get('pos'), "conf": msg.get('conf')})
        print(f"[BEACON] t={t:.2f} robot={msg.get('robot')} pos={msg.get('pos')} conf={msg.get('conf'):.2f}")
        
        # Track first victim discovery
        if first_victim_time is None:
            first_victim_time = t - start_time

    topics['beacon'].subscribe(beacon_handler)
    
    # Area explored handler: share negative information
    fully_explored_cells = set()
    def area_explored_handler(msg):
        cell = msg.get('cell')
        if cell:
            fully_explored_cells.add(cell)
    
    topics['area_explored'].subscribe(area_explored_handler)

    # Assign sectors to robots for coordinated dispersion
    # Divide grid into octants (8 sectors)
    sectors = []
    mid_x, mid_y = width // 2, height // 2
    quarter_x, quarter_y = width // 4, height // 4
    
    # Define 8 sectors (octants)
    sectors = [
        (0, mid_x, 0, mid_y),              # SW
        (mid_x, width, 0, mid_y),          # SE
        (0, mid_x, mid_y, height),         # NW
        (mid_x, width, mid_y, height),     # NE
        (quarter_x, mid_x + quarter_x, 0, mid_y),  # S-center
        (quarter_x, mid_x + quarter_x, mid_y, height),  # N-center
        (0, mid_x, quarter_y, mid_y + quarter_y),  # W-center
        (mid_x, width, quarter_y, mid_y + quarter_y)  # E-center
    ]

    robots = []
    for i in range(robots_count):
        # Assign sector (cycle through available sectors)
        sector = sectors[i % len(sectors)] if len(sectors) > 0 else None
        r = Robot(i, env, pheromap, topics, start_pos=(0, 0), comm_range=8, 
                 alpha=0.5, beta=3.0, energy=500, sector=sector, total_steps=steps)
        robots.append(r)

    # subscribe to status to detect failures
    failed_robots = set()
    def status_handler(msg):
        if msg.get('failed'):
            failed_robots.add(msg['robot'])

    topics['status'].subscribe(status_handler)
    
    # Robot position broadcasting for spreading pressure
    def broadcast_positions():
        positions = [(r.x, r.y) for r in robots if not r.failed]
        for r in robots:
            if not r.failed:
                r.nearby_robots = [pos for pos in positions if pos != (r.x, r.y)]
    
    # Share fully explored areas with all robots
    def share_explored_areas():
        for r in robots:
            if not r.failed:
                r.fully_explored_areas = fully_explored_cells.copy()

    for r in robots:
        r.start()

    # Intelligent pheromone seeding: create a weak "scent trail" outward from base
    # This guides robots to explore in all directions from (0,0) efficiently
    for radius in range(1, min(15, width // 3)):  # extend further for guidance
        for angle_steps in range(16):  # 16 directions for better coverage
            x = int(radius * math.cos(2 * math.pi * angle_steps / 16))
            y = int(radius * math.sin(2 * math.pi * angle_steps / 16))
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            if not env.is_obstacle(x, y):
                # pheromone strength decreases with distance: encourages spreading but weakens over distance
                strength = max(0.1, 1.0 / (1.0 + 0.1 * radius))
                pheromap.deposit(x, y, strength)

    viz = Visualizer(env, pheromap, robots, confirmed_victims=confirmed_victims)

    # metrics tracking
    total_free_cells = int((~env.obstacles).sum())
    initial_victim_count = int(env.victims.sum())
    coverage_history = []
    step_delay = 0.001  # reduced for faster execution
    et_step = None

    # Enhanced metrics tracking
    time_to_first_victim = None
    redundant_visit_history = []
    energy_history = []
    
    for step in range(steps):
        # Adaptive global pheromone evaporation
        pheromap.evaporate(rho=0.02, adaptive=True)
        
        # Broadcast robot positions for spreading pressure (every 5 steps)
        if step % 5 == 0:
            broadcast_positions()
        
        # Share fully explored areas (every 20 steps)
        if step % 20 == 0:
            share_explored_areas()

        # compute coverage and other metrics snapshot
        # aggregate visited counts per cell
        visit_counts = {}
        for r in robots:
            for cell in getattr(r, 'visited', set()):
                visit_counts[cell] = visit_counts.get(cell, 0) + 1

        explored_cells = len(visit_counts)
        coverage = explored_cells / max(1, total_free_cells)
        coverage_history.append(coverage)
        
        # Track redundant visits (cells visited >3 times)
        redundant_visits = sum(1 for count in visit_counts.values() if count > 3)
        redundant_visit_history.append(redundant_visits)
        
        # Track average remaining energy
        active_robots = [r for r in robots if not r.failed]
        if active_robots:
            avg_energy = sum(r.energy for r in active_robots) / len(active_robots)
            energy_history.append(avg_energy)

        # determine ET: time to reach final coverage plateau or detect all victims
        # we'll set ET later once we know final coverage, but also watch for all victims detected
        all_detected = False
        detected_by_swarm = set()
        for r in robots:
            detected_by_swarm.update(r.detected_victims)
        # update shared confirmed_victims set in-place so visualizer sees it
        confirmed_victims.clear()
        confirmed_victims.update(detected_by_swarm)
        if len(detected_by_swarm) >= initial_victim_count and et_step is None:
            et_step = step

        # visualize occasionally (reduced frequency for performance)
        if step % 50 == 0:
            viz.draw()
        time.sleep(step_delay)

        # simulate nearby robots inheriting tasks when failure happens
        if failed_robots:
            # naive handling: robots deposit extra pheromone at failed positions
            for fid in list(failed_robots):
                # find position from robot
                for r in robots:
                    if r.rid == fid:
                        pheromap.deposit(r.x, r.y, 2.0)
                        failed_robots.remove(fid)
                        break

    # stop robots
    for r in robots:
        r.stop()
    # join threads
    for r in robots:
        r.join(timeout=0.5)

    print('Simulation finished')
    # compute final metrics
    # final aggregated visit counts
    final_visit_counts = {}
    for r in robots:
        for cell in getattr(r, 'visited', set()):
            final_visit_counts[cell] = final_visit_counts.get(cell, 0) + 1

    explored_cells = len(final_visit_counts)
    CE = 100.0 * explored_cells / max(1, total_free_cells)

    # ET: if et_step recorded (all victims found), use that; otherwise, find first step where coverage reached final coverage
    final_coverage = coverage_history[-1] if coverage_history else (explored_cells / max(1, total_free_cells))
    ET_step = None
    for idx, c in enumerate(coverage_history):
        if abs(c - final_coverage) < 1e-9:
            ET_step = idx
            break
    if ET_step is None:
        ET_step = len(coverage_history) - 1
    ET = ET_step * step_delay

    # RR: overlapped area / total explored area
    overlapped = sum(1 for v in final_visit_counts.values() if v > 1)
    RR = 100.0 * overlapped / max(1, explored_cells)

    # EU: partition into zones and compute std deviation of coverage fractions
    zone_w = max(1, width // 6)
    zone_h = max(1, height // 4)
    zone_fracs = []
    for zy in range(0, height, zone_h):
        for zx in range(0, width, zone_w):
            free_in_zone = 0
            visited_in_zone = 0
            for yy in range(zy, min(zy + zone_h, height)):
                for xx in range(zx, min(zx + zone_w, width)):
                    if not env.obstacles[yy, xx]:
                        free_in_zone += 1
                        if (xx, yy) in final_visit_counts:
                            visited_in_zone += 1
            if free_in_zone > 0:
                zone_fracs.append(visited_in_zone / free_in_zone)
    import statistics
    EU = statistics.pstdev(zone_fracs) if zone_fracs else 0.0

    # Enhanced metrics
    avg_final_energy = sum(r.energy for r in robots if not r.failed) / max(1, len([r for r in robots if not r.failed]))
    energy_efficiency = (avg_final_energy / 500.0) * 100.0  # percentage of energy remaining
    
    # Redundant visits with threshold >3
    highly_redundant = sum(1 for v in final_visit_counts.values() if v > 3)
    highly_redundant_rate = 100.0 * highly_redundant / max(1, explored_cells)
    
    metrics = {
        'CoverageEfficiency_percent': CE,
        'ExplorationTime_seconds': ET,
        'RedundancyRate_percent': RR,
        'ExplorationUniformity_std': EU,
        'explored_cells': explored_cells,
        'total_free_cells': total_free_cells,
        'overlapped_cells': overlapped,
        'final_coverage_fraction': final_coverage,
        'TimeToFirstVictim_seconds': first_victim_time if first_victim_time else ET,
        'EnergyEfficiency_percent': energy_efficiency,
        'HighlyRedundantVisits_percent': highly_redundant_rate,
        'victims_detected': len(confirmed_victims),
        'total_victims': initial_victim_count,
    }

    # write metrics to files in project root
    import json, os
    out_json = os.path.join(os.getcwd(), 'metrics.json')
    out_txt = os.path.join(os.getcwd(), 'metrics.txt')
    with open(out_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(out_txt, 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print(f"Wrote metrics to {out_json} and {out_txt}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--robots', type=int, default=4)
    parser.add_argument('--width', type=int, default=60)
    parser.add_argument('--height', type=int, default=36)
    args = parser.parse_args()
    run_simulation(steps=args.steps, robots_count=args.robots, width=args.width, height=args.height)
