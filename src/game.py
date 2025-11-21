import os
import json
import random
import pygame
from pygame import KEYDOWN, KEYUP, K_UP, K_DOWN, K_LEFT, K_RIGHT
from pygame.locals import QUIT

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((100, 100))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.snake = [(5, 5)]
        self.food = (random.randint(0, 9), random.randint(0, 9))
        self.direction = K_RIGHT