from env import Connect6Env
import random

env = Connect6Env()

obs = env.reset()
done = False

while not done:
    action = random.randint(0, 360)
    next_obs, reward, done, info = env.step(action)

    
