import pygame as pg
import numpy as np


class Connect6Env():
    def __init__(self):
        # pygame
        pg.init()
        self.screen = pg.display.set_mode([500, 500])
        pg.display.set_caption("Connect6 Reinforcement Learning")
        self.clock = pg.time.Clock()
        self.clock.tick(10)

        # state
        self.state = np.zeros((19, 19), dtype=np.int)


    def reset(self):
        obs = self.state
        return obs

    
    def step_by_other(self, action):
        idx = (action // 19, action % 19)
        self.state[idx] = -1.0

    
    def finish_check(self):
        pass


    def step(self, action):
        idx = (action // 19, action % 19)

        if self.state[idx] == 0.0:
            self.state[idx] = 1.0
            reward = 0.0001
            info = {"pass":True}
        else:
            reward = -0.1
            info = {"pass":False}

        next_obs = self.state

        if self.finish_check():
            done = True
            reward += 10
        else:
            done = False

        return next_obs, reward, done, info


    def update(self):
        pass

