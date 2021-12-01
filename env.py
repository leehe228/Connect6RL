import pygame as pg
import numpy as np


class Connect6EnvAdversarial():
    def __init__(self) -> None:
        self.state = {0 : np.zeros((19, 19), dtype=np.int), 1 : np.zeros((19, 19), dtype=np.int)}
        self.reward_dict = {"victory" : 10.0, "defeat" : -10.0, "step" : 0.0001, "overlap" : -0.1}


    def reset(self):
        self.state = {0 : np.zeros((19, 19), dtype=np.int), 1 : np.zeros((19, 19), dtype=np.int)}
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


    def heuristic(self, turn : int, layer : int=0):
        """
        returns heuristic information about states
        turn : turn of agnet (0 or 1)
        layer : positive layer only (1), negative layer only (-1), both (0)
        """
        hlayer1 = np.zeros((19, 19), dtype=np.int)
        hlayer2 = np.zeros((19, 19), dtype=np.int)

        for i in range(0, 19):
            for j in range(0, 19):
                # init
                hlayer1[i, j] = 0
                hlayer2[i, j] = 0

                # left to right
                count1 = 0
                count2 = 0
                for k in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
                    try: 
                        if (1 == self.state[turn][i + k, j]): count1 += 1 
                    except: pass
                    try: 
                        if (-1 == self.state[turn][i + k, j]) : count2 += 1 
                    except: pass

                hlayer1[i, j] += count1
                hlayer2[i, j] += count2

                # up to down
                count1 = 0
                count2 = 0
                for k in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
                    try: 
                        if (1 == self.state[turn][i, j + k]): count1 += 1 
                    except: pass
                    try: 
                        if (-1 == self.state[turn][i, j + k]) : count2 += 1 
                    except: pass

                hlayer1[i, j] += count1
                hlayer2[i, j] += count2

                # diag upper left to bottom right \
                count1 = 0
                count2 = 0
                for k in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
                    try: 
                        if (1 == self.state[turn][i + k, j - k]): count1 += 1
                    except: pass
                    try: 
                        if (-1 == self.state[turn][i + k, j - k]) : count2 += 1 
                    except: pass

                hlayer1[i, j] += count1
                hlayer2[i, j] += count2


                # diag upper right to bottom left /
                count1 = 0
                count2 = 0
                for k in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
                    try: 
                        if (1 == self.state[turn][i - k, j + k]): count1 += 1
                    except: pass
                    try: 
                        if (-1 == self.state[turn][i - k, j + k]) : count2 += 1 
                    except: pass

                hlayer1[i, j] += count1
                hlayer2[i, j] += count2

        if layer == 1: return hlayer1
        elif layer == -1: return hlayer2
        else: return hlayer1, hlayer2


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
                info = {"pass" : True}
            else:
                reward = self.reward_dict["overlap"]
                info = {"pass" : False}

        else:
            if self.state[1][idx] == 0.0:
                self.state[1][idx] = 1.0
                self.state[0][idx] = -1.0
                reward = self.reward_dict["step"]
                info = {"pass" : True}
            else:
                reward = self.reward_dict["overlap"]
                info = {"pass" : False}

        next_obs = self.get_state(turn)

        w = self.finish_check()

        if w != 0 and (not info["pass"]):
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
