#-*-coding:utf-8-*-
import numpy as np
import random
import datetime
import time
import tensorflow as tf
from collections import deque
from env import Connect6EnvAdversarial

from mcts import Board, mcts_go
import copy

state_size = [15, 15]
action_size = 15 * 15

load_model = False
train_mode = True

batch_size = 64
mem_maxlen = 50000
discount_factor = 1.0
learning_rate = 0.0002

run_episode = 30000
test_episode = 100

max_step = 226

start_train_episode = 1000

target_update_step = 25
print_interval = 1
save_interval = 500

epsilon_init = 0.95
epsilon_min = 0.05

date_time = datetime.datetime.now().strftime("%d-%H-%M")

save_path = "./saved_models/"+ date_time + "_DQN_MCTS"
load_path = "./saved_models/"

class Model():
    def __init__(self, model_name):
        self.input = tf.placeholder(shape=[None, 1, state_size[0], state_size[1]], dtype=tf.float32)

        with tf.variable_scope(name_or_scope=model_name):
            self.conv1 = tf.layers.conv2d(self.input, 32, [6, 6], padding='SAME', activation=tf.nn.relu)
            self.pool1 = tf.layers.max_pooling2d(self.conv1, [6, 6], [1, 1], padding='SAME')
            self.conv2 = tf.layers.conv2d(self.pool1, 64, [6, 6], padding='SAME', activation=tf.nn.relu)
            self.pool2 = tf.layers.max_pooling2d(self.conv2, [4, 4], [1, 1], padding='SAME')
            # self.conv3 = tf.layers.conv2d(self.pool2, 64, [6, 6], padding='SAME', activation=tf.nn.relu)
            # self.pool3 = tf.layers.max_pooling2d(self.conv3, [4, 4], [1, 1], padding='SAME')
 
            self.flat = tf.layers.flatten(self.pool2)

            self.fc1 = tf.layers.dense(self.flat, 128, activation=tf.nn.relu)
            # self.fc2 = tf.layers.dense(self.flat, 512, activation=tf.nn.relu)
            self.Q_Out = tf.layers.dense(self.fc1, action_size, activation=tf.nn.softmax)

        self.predict = tf.argmax(self.Q_Out, 1)

        self.target_Q = tf.placeholder(shape=[None, action_size], dtype=tf.float32)

        self.loss = tf.losses.huber_loss(self.target_Q, self.Q_Out)
        self.UpdateModel = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model_name)


class DQNAgent():
    def __init__(self):
        self.model1 = Model("Q1")
        self.target_model1 = Model("target1")
        self.model2 = Model("Q2")
        self.target_model2 = Model("target2")

        self.memory1 = deque(maxlen=mem_maxlen)
        self.memory2 = deque(maxlen=mem_maxlen)

        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self.epsilon = epsilon_init

        self.Saver = tf.train.Saver()
        self.Summary, self.Merge = self.Make_Summary()

        if load_model == True:
            self.Saver.restore(self.sess, load_path)

        self.init_mcts()

    def init_mcts(self):
        self.game = Board(state_size[0])

    def set_mcts(self, action, turn):
        turn = -1 if turn == 0 else 1
        self.game.move(row=action // state_size[0], col=action % state_size[0], piece=turn)


    def get_action(self, state, turn : int):
        if self.epsilon > np.random.rand():
            # random_action = np.random.randint(0, action_size)
            # return random_action
            turn = -1 if turn == 0 else 1
            move = mcts_go(current_game=copy.deepcopy(self.game), team=turn, stats=True)
            action = move[0] * state_size[0] + move[1]
            return action
        else:
            if turn == 0:
                predict1 = self.sess.run(self.model1.predict, feed_dict={self.model1.input: [[state]]})
                return np.asscalar(predict1)
            else:
                predict2 = self.sess.run(self.model2.predict, feed_dict={self.model2.input: [[state]]})
                return np.asscalar(predict2)

    def append_sample(self, data, turn : int):
        if turn == 0:
            self.memory1.append(([data[0]], data[1], data[2], [data[3]], data[4]))
        else:
            self.memory2.append(([data[0]], data[1], data[2], [data[3]], data[4]))

    def save_model(self):
        self.Saver.save(self.sess, save_path + "/model/model")

    def train_model(self, model, target_model, memory, done):
        if done:
            if self.epsilon > epsilon_min:
                self.epsilon -= (0.5 / (run_episode - start_train_episode)) * 15

        mini_batch = random.sample(memory, batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for i in range(batch_size):
            states.append(mini_batch[i][0])
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states.append(mini_batch[i][3])
            dones.append(mini_batch[i][4])

        target = self.sess.run(model.Q_Out, feed_dict={model.input: states})
        target_val = self.sess.run(target_model.Q_Out, 
                                    feed_dict={target_model.input: next_states})

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + discount_factor * np.amax(target_val[i])

        _, loss = self.sess.run([model.UpdateModel, model.loss],
                                feed_dict={model.input: states, 
                                           model.target_Q: target})

        return loss

    def update_target(self, model, target_model):
        for i in range(len(model.trainable_var)):
            self.sess.run(target_model.trainable_var[i].assign(model.trainable_var[i]))

    def Make_Summary(self):
        self.summary_end_step = tf.placeholder(dtype = tf.float32)
        self.summary_mean_rewards = tf.placeholder(dtype=tf.float32)
        self.summary_max_rewards = tf.placeholder(dtype=tf.float32)
        self.summary_loss1 = tf.placeholder(dtype=tf.float32)
        self.summary_reward1 = tf.placeholder(dtype=tf.float32)
        self.summary_loss2 = tf.placeholder(dtype=tf.float32)
        self.summary_reward2 = tf.placeholder(dtype=tf.float32)
        
        tf.summary.scalar("end_step", self.summary_end_step)
        tf.summary.scalar("mean_rewards", self.summary_mean_rewards)
        tf.summary.scalar("max_rewards", self.summary_max_rewards)
        tf.summary.scalar("loss1", self.summary_loss1)
        tf.summary.scalar("reward1", self.summary_reward1)
        tf.summary.scalar("loss2", self.summary_loss2)
        tf.summary.scalar("reward2", self.summary_reward2)

        Summary = tf.summary.FileWriter(logdir=save_path, graph=self.sess.graph)
        Merge = tf.summary.merge_all()

        return Summary, Merge
        
    def Write_Summray(self, end_step, rewards, max_reward, reward1, loss1, reward2, loss2, episode):
        self.Summary.add_summary(
                self.sess.run(self.Merge, feed_dict={self.summary_end_step: end_step,
                                                self.summary_mean_rewards: rewards,
                                                self.summary_max_rewards: max_reward,
                                                self.summary_loss1: loss1, 
                                                self.summary_reward1: reward1, 
                                                self.summary_loss2: loss2, 
                                                self.summary_reward2: reward2}), episode)


def printBoard(state):
    print('-' * 38)
    for i in range(19):
        for j in range(19):
            if state[i, j] == 1.0:
                print("?óè ", end='')
            elif state[i, j] == -1.0:
                print("?óã ", end='')
            else: print("  ", end='')
        print()
    print('-' * 38)

if __name__ == '__main__':

    env = Connect6EnvAdversarial()
    agent = DQNAgent()

    rewards = {0 : [], 1 : []}
    losses = {0 : [], 1 : []}

    end_step = []

    for episode in range(run_episode + test_episode):
        if episode == run_episode:
            train_mode = False
        
        init_state = env.reset()
        states = {0 : init_state, 1 : init_state}
        dones = {0 : False, 1 : False}

        episode_rewards = {0 : 0.0, 1 : 0.0}

        first_turn = True
        
        for step in range(0, max_step * 2):

            print(f"episode : {episode}, step : {step // 2}", end='\r')

            turn = step % 2
            puts = 0

            while True:
                puts += 1

                while True:
                    action = agent.get_action(states[turn], turn)
                    next_state, reward, done, info = env.step(action, turn)

                    # print(next_state, reward, done, info)

                    if turn == 0:
                        os.system('clear')
                        printBoard(next_state)
                        print(f"epsidoe: {episode} / step : {step}\nact: ({action//19}, {action%19})\nreward: {round(reward, 4)}\ncum reward : {round(episode_rewards[0], 4)}:{round(episode_rewards[1], 4)} \ndone: {done} / info: {info}")

                    episode_rewards[turn] += reward
                    dones[turn] = done

                    if train_mode:
                        data = [states[turn], action, reward, next_state, done]
                        agent.append_sample(data, turn)
                    else:
                        agent.epsilon = 0.0

                    if info['pass'] or done: 
                        agent.set_mcts(action=action, turn=turn)
                        break
 
                states[turn] = next_state

                if episode > start_train_episode and train_mode:
                    # train behavior networks
                    if turn == 0:
                        loss1 = agent.train_model(agent.model1, agent.target_model1, agent.memory1, dones[0])
                        losses[0].append(loss1)
                    else:
                        loss2 = agent.train_model(agent.model2, agent.target_model2, agent.memory2, dones[1])
                        losses[1].append(loss2)

                    # update target networks
                    if step % (target_update_step) == 0:
                        if turn == 0:
                            agent.update_target(agent.model1, agent.target_model1)
                        else:
                            agent.update_target(agent.model2, agent.target_model2)

                if (first_turn): 
                    first_turn = False
                    break
                if puts == 2:
                    break
                if done:
                    break

            if dones[0] or dones[1]:
                break

        end_step.append(step)

        rewards[0].append(episode_rewards[0])
        rewards[1].append(episode_rewards[1])

        if episode % print_interval == 0 and episode != 0:
            print()
            print("step: {} / episode: {} / epsilon: {:.3f}".format(step, episode, agent.epsilon))
            print("reward: {:.2f} / reward1: {:.2f} / loss1: {:.4f} / reward2: {:.2f} / loss2: {:.4f}".format(
                  np.mean(rewards[0]) + np.mean(rewards[1]), np.mean(rewards[0]), np.mean(losses[0]), np.mean(rewards[1]), np.mean(losses[1])))
            print('------------------------------------------------------------')
            
            agent.Write_Summray(np.mean(end_step), np.mean(rewards[0]) + np.mean(rewards[1]), max(max(rewards[0]), max(rewards[1])), np.mean(rewards[0]), np.mean(losses[0]), 
                                np.mean(rewards[1]), np.mean(losses[1]), episode)

            rewards = {0 : [], 1 : []}
            losses = {0 : [], 1 : []}

        # ?\84\A4?\8A\B8?\9B\8C?\81\AC Î™®Îç∏ ????\9E\A5 
        if episode % save_interval == 0 and episode != 0:
            agent.save_model()
            print("Save Model {}".format(episode))

    env.close()
