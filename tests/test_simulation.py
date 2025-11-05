import threading
import time
from src.simulation.main import run_simulation


def test_short_run():
    # run a very short simulation to ensure no runtime errors
    run_simulation(steps=20, robots_count=3, width=30, height=18)

if __name__ == '__main__':
    test_short_run()
    print('Test run complete')
