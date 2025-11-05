# Swarm ACO Search-and-Rescue Simulation

This repository contains a lightweight Python simulation of a swarm of robots performing ACO-based exploration for search-and-rescue, including:

- simulated LIDAR/thermal/sound sensing
- behavior-based obstacle avoidance
- victim detection and pheromone beaconing
- ACO-based movement using pheromone (τ) and heuristic (η)
- pheromone deposition and evaporation
- simulated ROS-like topic pub/sub for map merging and broadcasts
- simulated communication loss and robot failure handing

This is intended as an educational simulator, not for real robots or ROS runtime.

Requirements
- Python 3.8+
- See `requirements.txt` for pip installable packages.

Quick run (PowerShell):

```powershell
python -m src.simulation.main --steps 200 --robots 6
```

Files of interest
- `src/simulation/main.py` - runner/CLI
- `src/simulation/environment.py` - grid, obstacles, victims
- `src/simulation/robot.py` - robot logic (sensing, ACO, behaviors)
- `src/simulation/pheromone_map.py` - shared pheromone map
- `src/simulation/communications.py` - simple topic pub/sub (simulated ROS)
- `src/simulation/visualize.py` - matplotlib visualization

See the code for parameters you can tune: pheromone evaporation (rho), alpha/beta, Q deposit, communication range.

License: MIT
