import greedy_snake
import torch
import random
import numpy as np
import pygame
import time


class GreedySnakeEnv:
    def __init__(self, cell_size=20, width_num=20, height_num=20, is_no_screen=False):
        self.game = greedy_snake.GreedySnake(cell_size, width_num, height_num, is_no_screen)
        self.state = None
        self.explored_area = []

    def get_state(self):
        state = np.zeros(12)
        if self.game.direction == "Up":
            state[4:8] = [1, 0, 0, 0]
        elif self.game.direction == "Down":
            state[4:8] = [0, 1, 0, 0]
        elif self.game.direction == "Left":
            state[4:8] = [0, 0, 1, 0]
        elif self.game.direction == "Right":
            state[4:8] = [0, 0, 0, 1]

        if self.game.snake_list[0][0] < self.game.food_pos[0]:
            state[11] = 1
        if self.game.snake_list[0][0] > self.game.food_pos[0]:
            state[10] = 1
        if self.game.snake_list[0][1] < self.game.food_pos[1]:
            state[9] = 1
        if self.game.snake_list[0][1] > self.game.food_pos[1]:
            state[8] = 1

        if ([self.game.snake_list[0][0], self.game.snake_list[0][1] - 1] in self.game.snake_list[1:]
        or self.game.snake_list[0][1] - 1 < 0):
            state[0] = 1
        if ([self.game.snake_list[0][0], self.game.snake_list[0][1] + 1] in self.game.snake_list[1:]
        or self.game.snake_list[0][1] + 1 >= self.game.height_num):
            state[1] = 1
        if ([self.game.snake_list[0][0] - 1, self.game.snake_list[0][1]] in self.game.snake_list[1:]
        or self.game.snake_list[0][0] - 1 < 0):
            state[2] = 1
        if ([self.game.snake_list[0][0] + 1, self.game.snake_list[0][1]] in self.game.snake_list[1:]
        or self.game.snake_list[0][0] + 1 >= self.game.width_num):
            state[3] = 1
        return torch.tensor(state).float()

    def step(self, action):
        if action == 0:
            self.game.direction = "Up"
        elif action == 1:
            self.game.direction = "Down"
        elif action == 2:
            self.game.direction = "Left"
        elif action == 3:
            self.game.direction = "Right"
        self.game.update_snake()
        self.game.check_game_over()
        if self.game.state == "Game Over":
            done = True
            self.game.reset()
            self.game.state = "Game Start"
        else:
            done = False
        if self.game.get_apple:
            reward = 10
            self.game.get_apple = False
            self.game.food_pos = random.choice(self.game.non_snake)
        elif done:
            reward = -10
            self.reset()
        else:
            reward = 0
        if self.game.screen is not None:
            self.game.screen.fill((255, 255, 255))
            self.game.draw_food()
            self.game.draw_snake()
            pygame.display.update()
        return self.get_state(), torch.tensor(reward).float(), torch.tensor(done)

    def reset(self):
        self.game.reset()
        self.state = None
        self.explored_area = []

# test code
# env = GreedySnakeEnv(cell_size=20, width_num=20, height_num=20, is_no_screen=True)
# while True:
#     time.sleep(0.08)
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             exit()
#     action_chosen = random.randint(0, 3)
#     state = env.get_state()
#     reward, next_state, done = env.step(action_chosen)
#     print(next_state)
#     print(action_chosen)
#     print(reward)
