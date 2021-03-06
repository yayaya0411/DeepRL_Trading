import random
from collections import deque
import os
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# from utility.utils import Portfolio
from agents.portfolio import Portfolio


# reference:
# https://arxiv.org/pdf/1312.5602.pdf
class Agent(Portfolio):
    def __init__(self, state_dim, balance, is_eval=False, model_name=""):
        super().__init__(balance=balance)
        self.model_type = 'DQN'
        self.state_dim = state_dim
        self.action_dim = 2  # hold, buy, sell
        self.memory = deque(maxlen=50)
        self.buffer_size = 30

        self.gamma = 0.95
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_min = 0.01  # minimum exploration rate
        self.epsilon_decay = 0.995 # decrease exploration rate as the agent becomes good at trading
        self.is_eval = is_eval
        # self.model = load_model(os.path.join(f'saved_models',f'{model_name}_{self.state_dim}_dim.h5')) if is_eval else self.model()
        if self.is_eval: 
            self.model = load_model(os.path.join(f'saved_models',f'{model_name}_{self.state_dim}_dim_ep20.h5'))    
        else:
            self.model = self.model()
        # print(self.model().summary())

        self.tensorboard = TensorBoard(log_dir='./logs/DQN_tensorboard', update_freq=90)
        self.tensorboard.set_model(self.model)

    def model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(self.action_dim, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(lr=0.01))
        return model

    def reset(self):
        self.reset_portfolio()
        self.epsilon = 1.0 # reset exploration rate

    def remember(self, state, actions, reward, next_state, done):
        self.memory.append((state, actions, reward, next_state, done))

    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        options = self.model.predict(state)
        return np.argmax(options[0])

    def experience_replay(self):
        # retrieve recent buffer_size long memory
        batch = [self.memory[i] for i in range(len(self.memory) - self.buffer_size + 1, len(self.memory))]
        sample_index = random.sample(range(29),10)
        mini_batch = [batch[i] for i in sample_index]

        for state, actions, reward, next_state, done in mini_batch:
            if not done:
                Q_target_value = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            else:
                Q_target_value = reward
                
            next_actions = self.model.predict(state)
            next_actions[0][np.argmax(actions)] = Q_target_value
            # print(Q_target_value,next_actions,next_actions[0][np.argmax(actions)])
            history = self.model.fit(state, next_actions, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return history.history['loss'][0], mini_batch
