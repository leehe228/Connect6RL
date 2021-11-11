import pygame as pg
import numpy as np


class Connect6Env():
    def __init__(self):
        # pygame
        # pg.init()
        # self.screen = pg.display.set_mode([500, 500])
        # pg.display.set_caption("Connect6 Reinforcement Learning")
        # self.clock = pg.time.Clock()
        # self.clock.tick(10)

        # state
        self.state = np.zeros((19, 19), dtype=np.int)

        # reward dictionary
        self.reward_dict = {"victory" : 10.0, "defeat" : -10.0, "step" : 0.0001, "overlap" : -0.1}


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
            reward = self.reward_dict["step"]
            info = {"pass":True}
        else:
            reward = self.reward_dict["overlap"]
            info = {"pass":False}

        next_obs = self.state

        if self.finish_check():
            done = True
            reward += self.reward_dict["victory"]
        else:
            done = False

        return next_obs, reward, done, info


    def update(self):
        for k in self.state:
            print(*k)


class Connect6EnvAdversarial():
    def __init__(self) -> None:
        # state
        self.state = {0 : np.zeros((19, 19), dtype=np.int), 1 : np.zeros((19, 19), dtype=np.int)}

        # reward dictionary
        self.reward_dict = {"victory" : 10.0, "defeat" : -10.0, "step" : 0.0001, "overlap" : -0.1}


    def reset(self):
        obs = self.state[0]
        return obs

    
    def finish_check(self, turn : int) -> bool:
        if (turn == 0):
            pass

        else:
            pass


    def step(self, action : int, turn : int) -> tuple(np.ndarray, float, bool, dict):
        idx = (action // 19, action % 19)

        if (turn == 0):
            if self.state[0][idx] == 0.0:
                self.state[0][idx] = 1.0
                self.state[1][idx] = -1.0
                reward = self.reward_dict["step"]
                info = {"pass":True}
            else:
                reward = self.reward_dict["overlap"]
                info = {"pass":False}

        else:
            if self.state[1][idx] == 0.0:
                self.state[1][idx] = 1.0
                self.state[0][idx] = -1.0
                reward = self.reward_dict["step"]
                info = {"pass":True}
            else:
                reward = self.reward_dict["overlap"]
                info = {"pass":False}

        next_obs = self.state[turn]

        if self.finish_check(turn):
            done = True
            reward += self.reward_dict["victory"]
        else:
            done = False

        return next_obs, reward, done, info


    def update(self):
        for k in self.state:
            print(*k)