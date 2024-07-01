import torch.nn as nn
import torch.optim as optim
from greedy_snake_env import GreedySnakeEnv
from collections import deque
import torch
import random
import time
import pygame
import os
BATCH_SIZE = 1000
MAX_MEMORY = 100_000


class Trainer:
    def __init__(self, net, lr, gamma, epsilon):
        self.net = net
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.game_times = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.epsilon = epsilon

    def choose_action(self, state):
        if random.randint(0, 100) < self.epsilon:
            return random.randint(0, 3)
        else:
            state0 = state
            action_prediction = self.net(state0)
            action = torch.argmax(action_prediction).item()
            return action

    def train_step(self, state, action, reward, next_state, done):
        state0 = torch.stack(state)
        new_state = torch.stack(next_state)
        action = torch.tensor(action)
        reward = torch.stack(reward)
        output = self.net(state0)
        target = output.clone()
        if state0.dim() > 1:
            for i in range(len(done)):
                if not done[i]:
                    Q_new = reward[i] + self.gamma * max(self.net(new_state[i]))
                else:
                    Q_new = reward[i]
                target[action[i]] = Q_new
            self.optimizer.zero_grad()
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            choose_samples = self.memory
        else:
            choose_samples = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*choose_samples)
        self.train_step(states, actions, rewards, next_states, dones)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def save_model(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.net.state_dict(), file_name)

    def train_short_memory(self, state, action, reward, next_state, done):
        state0 = state
        next_state = next_state
        reward = reward
        output = self.net(state0)
        target = output.clone()
        print(output)
        if not done:
            Q_new = reward + self.gamma * max(self.net(next_state))
        else:
            Q_new = reward
        target[action] = Q_new
        print(target)
        self.optimizer.zero_grad()
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    net_ = nn.Sequential(nn.Linear(12, 256), nn.ReLU(), nn.Linear(256, 4))
    env = GreedySnakeEnv(is_no_screen=True)
    trainer = Trainer(net_, 0.001, 0.8, 80)
    record = 0
    done = False
    while True:
        is_ = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        state_old = env.get_state()
        print(state_old)
        action = trainer.choose_action(state_old)
        print("Action:", action)
        score = env.game.score
        state_new, reward, done = env.step(action)
        print("State_New:", state_new)
        print("Reward:", reward)
        print("Done:", done)
        if record < score:
            record = score
            trainer.save_model()
        trainer.remember(state_old, torch.tensor(action), reward, state_new, done)
        trainer.train_short_memory(state_old, action, reward, state_new, done)
        if done:
            env.reset()
            print("Game:", trainer.game_times, "Score:", score, "Record:", record)
            trainer.game_times += 1
            trainer.epsilon = 80 - 0.1 * trainer.game_times
            trainer.train_long_memory()
