from env import Connect6Env
import random
import time

env = Connect6Env()

obs = env.reset()
done = False

step = 0
cum_reward = 0.0

while not done:
    action = random.randint(0, 360)
    next_obs, reward, done, info = env.step(action)

    cum_reward += reward
    print(step)
    env.update()
    print("reward :", cum_reward, end='\n\n')
    
    step += 1

    time.sleep(1.0)

