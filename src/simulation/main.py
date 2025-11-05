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
        "status": Topic()
    }

    # confirmed victims set (shared with visualizer)
    confirmed_victims = set()

    # central map merge handler
    def handle_map(msg):
        # msg: {'robot': id, 'map': np.array}
        pheromap.merge(msg['map'])

    topics['map_merge'].subscribe(handle_map)

    # beacon subscriber: log detection events and record beacon messages
    beacon_events = []
    import time as _time
    def beacon_handler(msg):
        # msg: {"robot": id, "pos": (x,y), "conf": float}
        t = _time.time()
        beacon_events.append({"time": t, "robot": msg.get('robot'), "pos": msg.get('pos'), "conf": msg.get('conf')})
        print(f"[BEACON] t={t:.2f} robot={msg.get('robot')} pos={msg.get('pos')} conf={msg.get('conf')}")

    topics['beacon'].subscribe(beacon_handler)

    robots = []
    for i in range(robots_count):
        # all robots start from base (0,0)
        r = Robot(i, env, pheromap, topics, start_pos=(0, 0), comm_range=8, alpha=0.5, beta=3.0)
        robots.append(r)

    # subscribe to status to detect failures
    failed_robots = set()
    def status_handler(msg):
        if msg.get('failed'):
            failed_robots.add(msg['robot'])

    topics['status'].subscribe(status_handler)

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
    step_delay = 0.05
    et_step = None

    for step in range(steps):
        # global pheromone evaporation
        pheromap.evaporate(rho=0.02)

        # compute coverage and other metrics snapshot
        # aggregate visited counts per cell
        visit_counts = {}
        for r in robots:
            for cell in getattr(r, 'visited', set()):
                visit_counts[cell] = visit_counts.get(cell, 0) + 1

        explored_cells = len(visit_counts)
        coverage = explored_cells / max(1, total_free_cells)
        coverage_history.append(coverage)

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

        # visualize occasionally
        if step % 5 == 0:
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

    metrics = {
        'CoverageEfficiency_percent': CE,
        'ExplorationTime_seconds': ET,
        'RedundancyRate_percent': RR,
        'ExplorationUniformity_std': EU,
        'explored_cells': explored_cells,
        'total_free_cells': total_free_cells,
        'overlapped_cells': overlapped,
        'final_coverage_fraction': final_coverage,
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
