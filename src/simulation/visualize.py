import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


class Visualizer:
    def __init__(self, env, pheromap, robots, confirmed_victims=None):
        self.env = env
        self.pheromap = pheromap
        self.robots = robots
        # a shared set (or container) of confirmed victim coordinates (x,y)
        self.confirmed_victims = confirmed_victims if confirmed_victims is not None else set()
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))

    def draw(self):
        self.ax.clear()
        # set white background
        self.ax.set_facecolor('white')
        
        # pheromone heatmap
        pher = self.pheromap.copy()
        im = self.ax.imshow(pher, cmap='hot', origin='lower')

        # obstacles overlay
        obs_y, obs_x = np.where(self.env.obstacles)
        obs_sc = self.ax.scatter(obs_x, obs_y, c='black', s=6, marker='s', label='Obstacle')

        # victims: draw unconfirmed victims in cyan, confirmed in red (larger)
        # find all victim positions
        vic_y, vic_x = np.where(self.env.victims)
        unconfirmed_x = []
        unconfirmed_y = []
        for x, y in zip(vic_x, vic_y):
            if (x, y) in self.confirmed_victims:
                continue
            unconfirmed_x.append(x)
            unconfirmed_y.append(y)
        vic_sc = None
        if unconfirmed_x:
            vic_sc = self.ax.scatter(unconfirmed_x, unconfirmed_y, c='cyan', s=30, marker='*', label='Victim (unconfirmed)')

        # confirmed victims
        conf_x = [c[0] for c in self.confirmed_victims if 0 <= c[0] < self.env.width and 0 <= c[1] < self.env.height]
        conf_y = [c[1] for c in self.confirmed_victims if 0 <= c[0] < self.env.width and 0 <= c[1] < self.env.height]
        conf_sc = None
        if conf_x:
            conf_sc = self.ax.scatter(conf_x, conf_y, c='red', s=80, marker='*', label='Victim (confirmed)')

        # robots: active and failed
        active_x, active_y = [], []
        failed_x, failed_y = [], []
        for r in self.robots:
            if r.failed:
                failed_x.append(r.x)
                failed_y.append(r.y)
            else:
                active_x.append(r.x)
                active_y.append(r.y)
        rob_active_sc = None
        rob_failed_sc = None
        if active_x:
            rob_active_sc = self.ax.scatter(active_x, active_y, c='blue', s=30, marker='o', label='Robot (active)')
        if failed_x:
            rob_failed_sc = self.ax.scatter(failed_x, failed_y, c='red', s=40, marker='x', label='Robot (failed)')

        # set limits and title
        self.ax.set_xlim(-0.5, self.env.width - 0.5)
        self.ax.set_ylim(-0.5, self.env.height - 0.5)
        self.ax.set_title('Pheromone map and robots')

        # legend: use explicit handles for clarity
        handles = []
        handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=6, label='Robot (active)'))
        handles.append(Line2D([0], [0], marker='x', color='r', markersize=8, label='Robot (failed)'))
        handles.append(Line2D([0], [0], marker='*', color='c', markeredgecolor='c', markersize=10, label='Victim (unconfirmed)'))
        handles.append(Line2D([0], [0], marker='*', color='r', markeredgecolor='r', markersize=14, label='Victim (confirmed)'))
        handles.append(Line2D([0], [0], marker='s', color='k', markersize=6, label='Obstacle'))
        self.ax.legend(handles=handles, loc='upper right', fontsize='small')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
