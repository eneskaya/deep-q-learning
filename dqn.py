# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
import os
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import SGD

import wandb
from wandb.keras import WandbCallback
wandb.init(project="is-deep-q")

config = wandb.config
config.activation_f_output_layer = 'linear'
config.activation_f_hidden_layer = 'sigmoid'
config.game = 'CartPole-v1'
config.batch_size = 32
config.gamma = 0.95
config.epsilon = 1.0
config.epsilon_min = 0.01
config.epsilon_decay = 0.995
config.learning_rate = 0.002

EPISODES = 100


class DQNAgent:
    '''
        CartPole-v1 Env (https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py):

            Observation (State): 
                Type: Box(4)
                Num	Observation                 Min         Max
                0	Cart Position             -4.8            4.8
                1	Cart Velocity             -Inf            Inf
                2	Pole Angle                 -24 deg        24 deg
                3	Pole Velocity At Tip      -Inf            Inf

            Actions:
                Type: Discrete(2)
                Num	Action
                0	Push cart to the left
                1	Push cart to the right

            Reward:
                Reward is 1 for every step taken, including the termination step
            Starting State:
                All observations are assigned a uniform random value in [-0.05..0.05]
            Episode Termination:
                Pole Angle is more than 12 degrees
                Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
                Episode length is greater than 200
                Solved Requirements
                Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.


        Der DQN Agent wird initialisiert mit:
            - state_size
            - action_size

        Hyperparameter:
            - memory:
                Für den Memory Replay Mechanismus (siehe DQN Paper von Deepmind)
                deque steht für "double-ended queue", siehe https://docs.python.org/2/library/collections.html#collections.deque
                    Deques support thread-safe, memory efficient appends and pops from either 
                    side of the deque with approximately the same O(1) performance in either direction.
            - gamma:
                Discount Rate oder Discount Factor,
                    Faktor 0 würde bedeuten, dass der Agent "kurzsichtig handelt", sprich zukünftige Belohnungen
                    werden nicht mehr betrachtet, nur die aktuelle Belohnung.
                    Faktor 1 würde bedeuten
            - epsilon:
                Die Explorations Rate, je höher, desto mehr wird "ausprobiert", also random actions
            - epsilon_min:
                Wir wollen eine Mindest-Exploration gewährleisten
            - epsilon_decay:
                Der Wert, um den der epsilon in jedem Timestep verfällt
    '''

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

        # Input layer
        model.add(Dense(24, input_dim=self.state_size,
                        activation=config.activation_f_hidden_layer))
        # Hidden layer 1
        model.add(Dense(24, activation=config.activation_f_hidden_layer))

        # Output layer
        model.add(Dense(self.action_size,
                        activation=config.activation_f_output_layer))

        optimizer = Adam(lr=self.learning_rate)
        wandb.config.optimizer = 'adam'

        # optimizer = RMSProp(lr=self.learning_rate)
        # wandb.config.optimizer = 'rmsprop'

        # optimizer = SGD(lr=self.learning_rate)
        # wandb.config.optimizer = 'sgd'

        model.compile(loss='mse',
                      optimizer=optimizer)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Exploring
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # Exploiting
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    '''
        Hier findet das Training statt.
        
        Im 'minibatch' werden aus dem memory des Agenten eine bestimmte Menge (batch_size)
        an (s, a, r, n_s, done) Tupeln entnommen. Für jedes Tupel wird nun folgender Algorithmus angewendet:
            
            ...
            
            Falls kein Terminal State, berechne target für gegebenen 
                target = 
                    reward 
                        +
                            gamma (discount rate, Wie hoch soll der zeR gewichtet werden) 
                                * 
                                    maximal erwartete, zukünftige (next_state) Q-Wert (Action-Value)

            Setze target_f als prediction für den aktuellen state
            Und aktualisiere für gegebene (next_state, action) den Wert in target_f

            Benutze state als input und target_f als target für unser DQN, starte training (fit)

            Falls epsilon noch über Schwellenwert, reduziere auf neuen Wert (epsilon * epsilon_decay)
    '''

    def replay(self, batch_size, e):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
                # Expected max. future reward
                wandb.log({'q_val': target, 'episode': e})

            target_f = self.model.predict(state)
            # print("predicted", target_f)
            target_f[0][action] = target
            # print("expected future reward", target_f)
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
    # agent.load(wandb.restore("cartpole-dqn-wandb.h5").name)
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
                agent.replay(batch_size, e)
        # if e % 10 == 0:
        #     agent.save(os.path.join(wandb.run.dir,
        #                             "cartpole-dqn-wandb.h5"))
