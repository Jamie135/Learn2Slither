import pygame
import random
from pygame import KEYDOWN, K_UP, K_DOWN, K_LEFT, K_RIGHT


BLOCK_SIZE = 40


class Snake:
    def __init__(self, parent_screen, grid_size, length=3):
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
        """Draw the snake on the screen."""
        # Draw background grid (playable area) and walls, then snake segments
        self.draw_background_grid()
        self.draw_walls()
        self.draw_snake()

    def draw_snake(self):
        """Draw the snake on the screen."""
        # Draw the head of the snake (first segment)
        self.parent_screen.blit(self.head, (self.x[0], self.y[0]))
        # Draw the body of the snake(remaining segments)
        for i in range(1, self.length):
            self.parent_screen.blit(self.body, (self.x[i], self.y[i]))

    def draw_background_grid(self):
        """Draw a grid for the playable area."""
        light_grey = (200, 200, 200)
        white = (240, 240, 240)

        # Indices 0 and grid_size - 1 are reserved for walls.
        for row in range(1, self.grid_size - 1):
            for col in range(1, self.grid_size - 1):
                color = light_grey if (row + col) % 2 == 0 else white
                pygame.draw.rect(
                    self.parent_screen,
                    color,
                    (
                        col * BLOCK_SIZE,
                        row * BLOCK_SIZE,
                        BLOCK_SIZE,
                        BLOCK_SIZE,
                    ),
                )

    def draw_walls(self):
        """Draw the walls around the playable area."""
        # Draw top and bottom walls
        for i in range(self.grid_size):
            pygame.draw.rect(
                self.parent_screen,
                (100, 100, 100),
                (i * BLOCK_SIZE, 0, BLOCK_SIZE, BLOCK_SIZE),
            )
            pygame.draw.rect(
                self.parent_screen,
                (100, 100, 100),
                (
                    i * BLOCK_SIZE,
                    (self.grid_size - 1) * BLOCK_SIZE,
                    BLOCK_SIZE,
                    BLOCK_SIZE,
                ),
            )
        # Draw left and right walls
        for i in range(1, self.grid_size - 1):
            pygame.draw.rect(
                self.parent_screen,
                (100, 100, 100),
                (0, i * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
            )
            pygame.draw.rect(
                self.parent_screen,
                (100, 100, 100),
                (
                    (self.grid_size - 1) * BLOCK_SIZE,
                    i * BLOCK_SIZE,
                    BLOCK_SIZE,
                    BLOCK_SIZE,
                ),
            )

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
        """Initialise the game."""
        self.playable_grid_size = grid_size
        # Total grid, including the surrounding wall layer
        self.grid_size = grid_size + 2
        screen_size = (
            self.grid_size * BLOCK_SIZE,
            self.grid_size * BLOCK_SIZE,
        )

        pygame.init()
        pygame.display.set_caption("Snake Game")
        self.SCREEN_UPDATE = pygame.USEREVENT
        pygame.time.set_timer(self.SCREEN_UPDATE, 300)
        self.surface = pygame.display.set_mode(screen_size)
        # Snake and apple receive the total grid size (including walls)
        self.snake = Snake(self.surface, self.grid_size)
        self.greenApple = GreenApple(self.surface, self.grid_size)
        self.snake.draw()

    def play(self):
        """Move the snake and draw the apple."""
        self.snake.move()
        self.greenApple.draw()

    def run(self):
        """Run the game."""
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
