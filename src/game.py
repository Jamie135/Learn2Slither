import random
import pygame
from pygame import KEYDOWN, KEYUP, K_UP, K_DOWN, K_LEFT, K_RIGHT


SCREEN_SIZE = (400, 400)
BLOCK_SIZE = 40


class Snake:
    def __init__(self, parent_screen, length=3):
        self.length = length
        self.parent_screen = parent_screen
        # Use the existing snake image file for the head
        self.head = pygame.image.load("images/snake.png")
        self.body = pygame.image.load("images/snakebody.jpg")
        self.x = [BLOCK_SIZE] * length
        self.y = [BLOCK_SIZE] * length
        # to be randomised
        self.direction = "RIGHT"

    def draw(self):
        # to be randomised
        self.parent_screen.fill((0, 0, 0))
        for i in range(self.length):
            self.parent_screen.blit(self.body, (self.x[i], self.y[i]))

    def increase(self):
        self.length += 1
        self.x.append(-1)
        self.y.append(-1)

    def move_left(self):
        if self.direction != "RIGHT":
            self.direction = "LEFT"

    def move_right(self):
        if self.direction != "LEFT":
            self.direction = "RIGHT"

    def move_up(self):
        if self.direction != "DOWN":
            self.direction = "UP"

    def move_down(self):
        if self.direction != "UP":
            self.direction = "DOWN"

    def move(self):
        for i in range(self.length - 1, 0, -1):
            self.x[i] = self.x[i - 1]
            self.y[i] = self.y[i - 1]
        if self.direction == "RIGHT":
            self.x[0] += BLOCK_SIZE
        if self.direction == "LEFT":
            self.x[0] -= BLOCK_SIZE
        if self.direction == "UP":
            self.y[0] += BLOCK_SIZE
        if self.direction == "DOWN":
            self.y[0] -= BLOCK_SIZE
        self.draw()


class Apple:
    def __init__(self, parent_screen):
        self.parent_screen = parent_screen
        self.image = pygame.image.load("images/applegreen.png")
        self.x = BLOCK_SIZE
        self.y = BLOCK_SIZE

    def draw(self):
        self.parent_screen.blit(self.image, (self.x, self.y))


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Snake Game")
        self.SCREEN_UPDATE = pygame.USEREVENT
        pygame.time.set_timer(self.SCREEN_UPDATE, 300)
        self.surface = pygame.display.set_mode(SCREEN_SIZE)
        self.snake = Snake(self.surface)
        self.snake.draw()

    def play(self):
        self.snake.move()

    def run(self):
        running = True
        while running:
            self.surface.fill((0, 0, 0))
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == self.SCREEN_UPDATE:
                    self.play()
            # Update the display
            pygame.display.update()


game = Game()
game.run()
