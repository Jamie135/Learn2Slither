import pygame
import random
BLOCK_SIZE = 40


class GreenApple:
    def __init__(self, parent_screen, grid_size):
        self.parent_screen = parent_screen
        self.grid_size = grid_size
        self.image = pygame.image.load("images/applegreen.png")
        # Random position inside the play area (excluding walls)
        self.x = BLOCK_SIZE * random.randint(1, grid_size - 2)
        self.y = BLOCK_SIZE * random.randint(1, grid_size - 2)

    def draw(self):
        self.parent_screen.blit(self.image, (self.x, self.y))

    def randomize_position(self):
        # Randomize apple position inside play area
        self.x = BLOCK_SIZE * random.randint(1, self.grid_size - 2)
        self.y = BLOCK_SIZE * random.randint(1, self.grid_size - 2)
