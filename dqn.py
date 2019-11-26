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

EPISODES = 100


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        activation_f_output_layer = 'linear'
        activation_f_hidden_layer = 'sigmoid'

        wandb.config.activation_f_output_layer = activation_f_output_layer
        wandb.config.activation_f_hidden_layer = activation_f_hidden_layer

        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation=activation_f_hidden_layer))
        model.add(Dense(24, activation=activation_f_hidden_layer))
        model.add(Dense(self.action_size, activation=activation_f_output_layer))
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
    batch_size = 32

    wandb.config.game = game
    wandb.config.batch_size = batch_size
    wandb.config.state_size = state_size
    wandb.config.action_size = action_size
    wandb.config.gamma = agent.gamma
    wandb.config.epsilon = agent.epsilon
    wandb.config.epsilon_min = agent.epsilon_min
    wandb.config.epsilon_decay = agent.epsilon_decay
    wandb.config.learning_rate = agent.learning_rate

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
