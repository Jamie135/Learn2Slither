import random
import pygame
import argparse
from pygame import KEYDOWN, KEYUP, K_UP, K_DOWN, K_LEFT, K_RIGHT


BLOCK_SIZE = 40


class Snake:
    def __init__(self, parent_screen, grid_size, length=3):  # Fixed syntax error
        self.length = length
        self.parent_screen = parent_screen
        self.grid_size = grid_size
        # Use the existing snake image file for the head
        self.head = pygame.image.load("images/snake.png")
        self.body = pygame.image.load("images/snakebody.jpg")
        self.x = [BLOCK_SIZE * 2] * length  # Start inside the wall
        self.y = [BLOCK_SIZE * 2] * length
        # to be randomised
        self.direction = "RIGHT"

    def draw(self):
        self.parent_screen.fill((0, 0, 0))
        # Draw walls
        self.draw_walls()
        # Draw the head (first segment)
        self.parent_screen.blit(self.head, (self.x[0], self.y[0]))
        # Draw the body (remaining segments)
        for i in range(1, self.length):
            self.parent_screen.blit(self.body, (self.x[i], self.y[i]))

    def draw_walls(self):
        # Draw top and bottom walls
        for i in range(self.grid_size):
            pygame.draw.rect(self.parent_screen, (100, 100, 100), 
                           (i * BLOCK_SIZE, 0, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.parent_screen, (100, 100, 100), 
                           (i * BLOCK_SIZE, (self.grid_size - 1) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        # Draw left and right walls
        for i in range(1, self.grid_size - 1):
            pygame.draw.rect(self.parent_screen, (100, 100, 100), 
                           (0, i * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.parent_screen, (100, 100, 100), 
                           ((self.grid_size - 1) * BLOCK_SIZE, i * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

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
            self.y[0] -= BLOCK_SIZE
        if self.direction == "DOWN":
            self.y[0] += BLOCK_SIZE
        self.draw()


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


class Game:
    def __init__(self, grid_size):
        # Validate minimum grid size
        # Theoretical minimum: 6x6 (4x4 play area for 13 segments + 1 apple)
        # Practical minimum: 10x10 (8x8 play area for comfortable gameplay)
        if grid_size < 10:
            print("Error: Grid size must be at least 10x10 (8x8 play area + 2 walls)")
            print("Using minimum grid size of 10x10")
            grid_size = 10
        
        self.grid_size = grid_size
        screen_size = (grid_size * BLOCK_SIZE, grid_size * BLOCK_SIZE)
        
        pygame.init()
        pygame.display.set_caption("Snake Game")
        self.SCREEN_UPDATE = pygame.USEREVENT
        pygame.time.set_timer(self.SCREEN_UPDATE, 300)
        self.surface = pygame.display.set_mode(screen_size)
        self.snake = Snake(self.surface, grid_size)
        self.greenApple = GreenApple(self.surface, grid_size)
        self.snake.draw()

    def play(self):
        self.snake.move()
        self.greenApple.draw()

    def run(self):
        running = True
        while running:
            # Handle events
            for event in pygame.event.get():

                if event.type == KEYDOWN:
                    if event.key == K_UP:
                        self.snake.move_up()
                    if event.key == K_DOWN:
                        self.snake.move_down()
                    if event.key == K_LEFT:
                        self.snake.move_left()
                    if event.key == K_RIGHT:
                        self.snake.move_right()

                if event.type == pygame.QUIT:
                    running = False
                elif event.type == self.SCREEN_UPDATE:
                    self.play()
            # Update the display
            pygame.display.update()


if __name__ == "__main__":
    # Parse command line arguments using argparse
    parser = argparse.ArgumentParser(description='Snake Game - Learn2Slither')
    parser.add_argument(
        'grid_size',
        type=int,
        nargs='?',
        default=12,
        help='Grid size (minimum 10x10, default 12x12)'
    )
    
    args = parser.parse_args()
    
    game = Game(args.grid_size)
    game.run()
