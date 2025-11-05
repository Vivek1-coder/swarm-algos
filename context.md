# Project Context

## Current Goal
- Simulate an ant-colony-inspired multi-robot search-and-rescue swarm.
- Emphasize balanced exploration versus exploitation with every robot starting at the base cell `(0, 0)`.
- Track coverage efficiency, exploration timing, redundancy, and exploration uniformity after each run.

## Code Layout
- `src/simulation/main.py`: Entry point that seeds pheromones, launches robots, runs the simulation loop, and writes metrics.
- `src/simulation/robot.py`: Robot agent thread with sensing, avoidance, pheromone deposition, energy management, and ACO-driven motion.
- `src/simulation/environment.py`: Grid world generator with obstacles, victims, and neighbor queries for movement.
- `src/simulation/pheromone_map.py`: Thread-safe shared pheromone grid with deposit, evaporation, and merge support.
- `src/simulation/visualize.py`: Matplotlib visualizer with white background, heatmap pheromone overlay, robot markers, victim status, and legend.
- `src/simulation/communications.py`: Lightweight in-process pub/sub topics used for beacon broadcasts, status, and map sharing.
- `tests/test_simulation.py`: Smoke test that runs a short simulation to ensure basic stability.

## Key Parameters
- Grid defaults to 60×36 cells with roughly 7% obstacles and six random victims.
- ACO weights: `alpha=0.5`, `beta=3.0`, evaporation `rho=0.02`, pheromone deposit `Q=1.0`.
- Robots: default 12 agents, communication range 8, initial energy 500 with recharge threshold 50, failure probability `1e-5`.
- Heuristic boosts: unexplored cells +4.0, pheromone bonus `0.5 × tau`, plus distance-based bias for outward movement.

## Running The Simulation
- Activate the virtual environment if needed: `.\.venv\Scripts\Activate.ps1`.
- Execute: `python -m src.simulation.main --steps 3000 --robots 12` (adjust flags as required; see `main.py` for options).
- Live visualization opens every five steps; console logs `[BEACON]` entries when victims are detected.
- Metrics are written to `metrics.json` and `metrics.txt` in the workspace root.

## Recent Enhancements
- All robots now deploy from `(0, 0)` to mimic a shared staging area.
- Expanded radial pheromone seeding: 16 directions, 15-cell reach, strength decays by `1/(1+0.1*r)`.
- Reduced random walk jitter to 2% to favor informed moves.
- Confirmed victims highlighted in red stars within the visualizer, with beacon subscribers logging detections.

## Outstanding Work
- Implement richer map-merging logic and failure-handling scenarios on the communications layer.
- Strengthen automated tests with assertions on pheromone dynamics and coverage performance.
- Document full setup and usage instructions for Windows PowerShell in the README.
- Iterate on parameter tuning if coverage efficiency remains below target (current run ≈55.6%).
