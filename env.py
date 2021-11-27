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
        self.state = {0 : np.zeros((19, 19), dtype=np.int), 1 : np.zeros((19, 19), dtype=np.int)}
        self.reward_dict = {"victory" : 10.0, "defeat" : -10.0, "step" : 0.00001, "overlap" : -0.1}


    def reset(self):
        obs = self.state[0]
        return obs

    
    def finish_check(self) -> bool:
        for i in range(0, 19):
            for j in range(0, 19):
                for k in [1, -1]:
                    try: 
                        if (k == self.state[0][i, j] == self.state[0][i + 1, j] == self.state[0][i + 2, j] == self.state[0][i + 3, j] == self.state[0][i + 4, j] == self.state[0][i + 5, j]):
                            return k
                    except: pass

        for j in range(0, 19):
            for i in range(0, 19):
                for k in [1, -1]:
                    try: 
                        if (k == self.state[0][i, j] == self.state[0][i, j + 1] == self.state[0][i, j + 2] == self.state[0][i, j + 3] == self.state[0][i, j + 4] == self.state[0][i, j + 5]):
                            return k
                    except: pass

        for i in range(0, 19):
            for j in range(0, 19):
                for k in [1, -1]:
                    try: 
                        if (k == self.state[0][i, j] == self.state[0][i + 1, j + 1] == self.state[0][i + 2, j + 2] == self.state[0][i + 3, j + 3] == self.state[0][i + 4, j + 4] == self.state[0][i + 5, j + 5]):
                            return k
                    except: pass

        for i in range(0, 19):
            for j in range(0, 19):
                for k in [1, -1]:
                    try: 
                        if (k == self.state[0][i, j] == self.state[0][i - 1, j + 1] == self.state[0][i - 2, j + 2] == self.stat[0][i - 3, j + 3] == self.state[0][i - 4, j + 4] == self.state[0][i - 5, j + 5]):
                            return k
                    except: pass

        return 0


    def get_state(self, turn : int):
        return self.state[turn]

    
    def put_check(self, action : int, turn : int):
        idx = (action // 19, action % 19)

        if self.state[turn][idx] != 0.0:
            return False

        return True


    def step(self, action : int, turn : int):
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

        next_obs = self.get_state(turn)

        w = self.finish_check()

        if w != 0:
            done = True
            if (w == 1 and turn == 0) or (w == -1 and turn == 1):
                reward += self.reward_dict["victory"]
            else:
                reward += self.reward_dict["defeat"]
        else:
            done = False

        return next_obs, reward, done, info


    def update(self):
        for k in self.state:
            print(*k)
