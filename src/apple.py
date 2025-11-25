import pygame
import random

BLOCK_SIZE = 40


class Apple:
    def __init__(self, parent_screen, grid_size, occupied_positions, kind):
        """
        Initialise the apples.
        """
        self.parent_screen = parent_screen
        self.grid_size = grid_size
        self.kind = kind

        if kind == "green":
            image_path = "images/applegreen.png"
        else:
            image_path = "images/applered.png"

        self.image = pygame.image.load(image_path)
        # Place the apple at a random position inside the play area that does
        # not collide with any occupied position (snake or other apples).
        self.randomize_position(occupied_positions)

    @staticmethod
    def free_position(grid_size, occupied):
        """Return a random (x, y) inside the playable area not in occupied."""
        while True:
            x = BLOCK_SIZE * random.randint(1, grid_size - 2)
            y = BLOCK_SIZE * random.randint(1, grid_size - 2)
            if (x, y) not in occupied:
                return x, y

    @classmethod
    def spawn_apples(cls, parent_screen, grid_size, snake):
        """
        Spawn two green apples and one red apple in free cells.
        """
        # Initial occupied positions: all snake segments.
        occupied = {(snake.x[i], snake.y[i]) for i in range(snake.length)}

        green_apples = []
        for _ in range(2):
            apple = cls(parent_screen, grid_size, occupied, "green")
            green_apples.append(apple)
            occupied.add((apple.x, apple.y))

        red_apple = cls(parent_screen, grid_size, occupied, "red")
        return green_apples, red_apple

    def draw(self):
        """Draw the apples on the screen."""
        self.parent_screen.blit(self.image, (self.x, self.y))

    def randomize_position(self, occupied_positions):
        """Randomize apples position avoiding all occupied positions."""
        self.x, self.y = self.free_position(self.grid_size, occupied_positions)
