import pygame
import time
import random


class GreedySnake:
    def __init__(self, cell_size, width_num, height_num, is_no_screen=False):
        """
        :param cell_size: 每格的大小
        :param width_num: 宽有几格
        :param height_num: 高有几格
        :param is_no_screen: 是否需要弹出屏幕
        """
        self.cell_size = cell_size
        self.width_num = width_num
        self.height_num = height_num
        self.snake_list = [[self.width_num // 2, self.height_num // 2 + 1], [self.width_num // 2, self.height_num // 2], [self.width_num // 2, self.height_num // 2 - 1]]
        self.non_snake = [[i, j] for i in range(width_num) for j in range(height_num) if [i, j] not in self.snake_list]
        self.food_pos = random.choice(self.non_snake)
        self.direction = "Down"
        self.width = self.width_num * self.cell_size
        self.height = self.height_num * self.cell_size
        self.get_apple = False
        self.score = 0
        self.state = "Game Start"
        if is_no_screen:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
        else:
            self.screen = None

    def draw_snake(self):
        """
        画出蛇
        """
        for i in self.snake_list:
            pygame.draw.rect(self.screen, (0, 0, 255), (i[0] * self.cell_size + 1, i[1] * self.cell_size + 1, self.cell_size - 2, self.cell_size - 2), border_radius=5)

    def draw_food(self):
        """
        画出食物
        """
        pygame.draw.rect(self.screen, (255, 0, 0), (self.food_pos[0] * self.cell_size, self.food_pos[1] * self.cell_size, self.cell_size, self.cell_size))

    def update_snake(self):
        """
        更新蛇的信息
        """
        if self.direction == "Up":
            self.snake_list.insert(0, [self.snake_list[0][0], self.snake_list[0][1] - 1])
        elif self.direction == "Down":
            self.snake_list.insert(0, [self.snake_list[0][0], self.snake_list[0][1] + 1])
        elif self.direction == "Left":
            self.snake_list.insert(0, [self.snake_list[0][0] - 1, self.snake_list[0][1]])
        elif self.direction == "Right":
            self.snake_list.insert(0, [self.snake_list[0][0] + 1, self.snake_list[0][1]])
        if self.snake_list[0] != self.food_pos:
            self.snake_list.pop()
        else:
            self.get_apple = True
            self.score += 10
        self.non_snake = [[i, j] for i in range(self.width_num) for j in range(self.height_num) if [i, j] not in self.snake_list]

    def draw_welcome_screen(self):
        """
        欢迎画面
        """
        font = pygame.font.Font(None, 50)
        text = font.render("Welcome to Greedy Snake", True, (122, 122, 122))
        text_start = font.render("Press Space to start", True, (122, 122, 122))
        text_rect = text.get_rect()
        self.screen.blit(text, (self.width // 2 - text_rect.width // 2, self.height // 2 - text_rect.height // 2))
        text_rect = text_start.get_rect()
        self.screen.blit(text_start, (self.width // 2 - text_rect.width // 2, self.height // 2 + text_rect.height // 2 + 10))

    def draw_game_over_screen(self):
        """
        游戏结束画面
        """
        font = pygame.font.Font(None, 50)
        text = font.render("Game Over", True, (122, 122, 122))
        text_start = font.render("Press Space to restart", True, (122, 122, 122))
        text_score = font.render("Your Score: " + str(self.score), True, (122, 122, 122))
        text_rect = text.get_rect()
        self.screen.blit(text, (self.width // 2 - text_rect.width // 2, self.height // 2 - text_rect.height // 2))
        text_rect = text_start.get_rect()
        self.screen.blit(text_start, (self.width // 2 - text_rect.width // 2, self.height // 2 + text_rect.height // 2 + 10))
        text_rect = text_score.get_rect()
        self.screen.blit(text_score, (self.width // 2 - text_rect.width // 2, self.height // 2 + text_rect.height // 2 + 50))

    def check_game_over(self):
        """
        检查游戏是否结束
        """
        if self.snake_list[0][0] < 0 or self.snake_list[0][0] >= self.width_num or self.snake_list[0][1] < 0 or self.snake_list[0][1] >= self.height_num:
            self.state = "Game Over"
        elif self.snake_list[0] in self.snake_list[1:]:
            self.state = "Game Over"

    def reset(self):
        """
        重置游戏
        """
        self.snake_list = [[self.width_num // 2, self.height_num // 2 + 1], [self.width_num // 2, self.height_num // 2], [self.width_num // 2, self.height_num // 2 - 1]]
        self.non_snake = [[i, j] for i in range(self.width_num) for j in range(self.height_num) if [i, j] not in self.snake_list]
        self.food_pos = random.choice(self.non_snake)
        self.direction = "Down"
        self.score = 0


if __name__ == '__main__':
    start_time = time.time()
    is_no_screen_actual = True
    greedy_snake = GreedySnake(20, 20, 20, is_no_screen_actual)
    greedy_snake.food_pos = random.choice(greedy_snake.non_snake)
    if is_no_screen_actual:
        while True:
            if greedy_snake.get_apple:
                greedy_snake.food_pos = random.choice(greedy_snake.non_snake)
                greedy_snake.get_apple = False
            greedy_snake.screen.fill((255, 255, 255))
            if greedy_snake.state == "Game Start":
                greedy_snake.draw_welcome_screen()
            elif greedy_snake.state == "Game Running":
                greedy_snake.draw_food()
                greedy_snake.draw_snake()
                if time.time() - start_time > 0.1:
                    greedy_snake.update_snake()
                    start_time = time.time()
            greedy_snake.check_game_over()
            if greedy_snake.state == "Game Over":
                greedy_snake.draw_game_over_screen()
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] and greedy_snake.direction != "Down":
                greedy_snake.direction = "Up"
            elif keys[pygame.K_DOWN] and greedy_snake.direction != "Up":
                greedy_snake.direction = "Down"
            elif keys[pygame.K_LEFT] and greedy_snake.direction != "Right":
                greedy_snake.direction = "Left"
            elif keys[pygame.K_RIGHT] and greedy_snake.direction != "Left":
                greedy_snake.direction = "Right"
            if keys[pygame.K_SPACE] and (greedy_snake.state == "Game Start" or greedy_snake.state == "Game Over"):
                greedy_snake.state = "Game Running"
                greedy_snake.reset()
