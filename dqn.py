# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
import os
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import wandb
from wandb.keras import WandbCallback
wandb.init(project="is-deep-q")

config = wandb.config
config.activation_f_output_layer = 'linear'
config.activation_f_hidden_layer = 'relu'
config.game = 'CartPole-v1'
config.batch_size = 32
config.gamma = 0.95
config.epsilon = 1.0
config.epsilon_min = 0.01
config.epsilon_decay = 0.995
config.learning_rate = 0.001

EPISODES = 100


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)

        self.gamma = config.gamma    # discount rate
        self.epsilon = config.epsilon  # exploration rate
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.learning_rate = config.learning_rate

        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size,
                        activation=config.activation_f_hidden_layer))
        model.add(Dense(24, activation=config.activation_f_hidden_layer))
        model.add(Dense(self.action_size,
                        activation=config.activation_f_output_layer))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    '''
        Hier findet das Training statt.
        
        Im 'minibatch' werden aus dem memory des Agenten eine bestimmte Menge (batch_size)
        an (s, a, r, n_s, done) Tupeln entnommen. Für jedes Tupel wird nun folgender Algorithmus angewendet:
    '''
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0,
                           callbacks=[WandbCallback()])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    game = 'CartPole-v1'
    env = gym.make(game)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = config.batch_size

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for time in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                wandb.log({'episode': e, 'score': time,
                           'epsilon': agent.epsilon})
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save(os.path.join(wandb.run.dir,
        #                             "cartpole-dqn-wandb.h5"))
