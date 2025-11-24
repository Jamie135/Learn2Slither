import os
import json
import random
import pygame
from pygame import KEYDOWN, KEYUP, K_UP, K_DOWN, K_LEFT, K_RIGHT


SCREEN_SIZE = (400, 400)
BLOCK_SIZE = 40


class Snake:
    def __init__(self, parent_screen, length=3):
        self.length = length
        self.parent_screen = parent_screen
        self.head = pygame.image.load("images/snakehead.jpg")
        self.body = pygame.image.load("images/snakebody.jpg")
        self.x = [BLOCK_SIZE] * length
        self.y = [BLOCK_SIZE] * length
        self.direction = "RIGHT"


class Apple:
    pass


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Snake Game")
        self.SCREEN_UPDATE = pygame.USEREVENT
        pygame.time.set_timer(self.SCREEN_UPDATE, 300)
        self.surface = pygame.display.set_mode(SCREEN_SIZE)

    def run(self):
        running = True
        while running:
            self.surface.fill((0, 0, 0))
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Update the display
            pygame.display.update()


game = Game()
game.run()
