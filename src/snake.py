import pygame
import random

BLOCK_SIZE = 40


class Snake:
    def __init__(self, parent_screen, grid_size, length=3):
        """Initialise the snake."""
        self.length = length
        self.parent_screen = parent_screen
        self.grid_size = grid_size
        # Use the existing snake image file for the head
        self.head = pygame.image.load("images/snake.png")
        self.body = pygame.image.load("images/snakebody.jpg")
        self.position_snake()

    def position_snake(self):
        """Randomly place the snake contiguously on the playable field."""
        length = self.length
        playable_min = 1
        playable_max = self.grid_size - 2
        playable_span = playable_max - playable_min + 1

        if length <= playable_span:
            # Choose a random direction and place the snake contiguously
            # so that the head is at the front of the direction of travel.
            direction = random.choice(["RIGHT", "LEFT", "UP", "DOWN"])

            if direction == "RIGHT":
                # Head on the right, body extending left.
                row = random.randint(playable_min, playable_max)
                head_col = random.randint(
                    playable_min + length - 1,
                    playable_max,
                )
                self.x = [(head_col - i) * BLOCK_SIZE for i in range(length)]
                self.y = [row * BLOCK_SIZE for _ in range(length)]

            elif direction == "LEFT":
                # Head on the left, body extending right.
                row = random.randint(playable_min, playable_max)
                head_col = random.randint(
                    playable_min,
                    playable_max - length + 1,
                )
                self.x = [(head_col + i) * BLOCK_SIZE for i in range(length)]
                self.y = [row * BLOCK_SIZE for _ in range(length)]

            elif direction == "DOWN":
                # Head at the bottom, body extending up.
                col = random.randint(playable_min, playable_max)
                head_row = random.randint(
                    playable_min + length - 1,
                    playable_max,
                )
                self.x = [col * BLOCK_SIZE for _ in range(length)]
                self.y = [(head_row - i) * BLOCK_SIZE for i in range(length)]

            else:  # direction == "UP"
                # Head at the top, body extending down.
                col = random.randint(playable_min, playable_max)
                head_row = random.randint(
                    playable_min,
                    playable_max - length + 1,
                )
                self.x = [col * BLOCK_SIZE for _ in range(length)]
                self.y = [(head_row + i) * BLOCK_SIZE for i in range(length)]

            self.direction = direction
        else:
            # Fallback: place the snake along a serpentine path inside
            # the playable area so segments remain contiguous.
            path = []
            for row in range(playable_min, playable_max + 1):
                cols = range(playable_min, playable_max + 1)
                if (row - playable_min) % 2 == 1:
                    cols = reversed(list(cols))
                for col in cols:
                    path.append((col, row))

            if length > len(path):
                raise ValueError("Snake length is too long.")

            start_idx = random.randint(0, len(path) - length)
            segment = path[start_idx:start_idx + length]

            self.x = [col * BLOCK_SIZE for col, _ in segment]
            self.y = [row * BLOCK_SIZE for _, row in segment]

            if length >= 2:
                dx = segment[0][0] - segment[1][0]
                dy = segment[0][1] - segment[1][1]
                if dx == 1:
                    self.direction = "RIGHT"
                elif dx == -1:
                    self.direction = "LEFT"
                elif dy == 1:
                    self.direction = "DOWN"
                elif dy == -1:
                    self.direction = "UP"
            else:
                self.direction = random.choice(["RIGHT", "LEFT", "UP", "DOWN"])
        # Ensure the first move will not immediately hit a wall by
        # shifting the whole snake one cell inward if necessary.
        self.avoid_collision()

    def avoid_collision(self):
        """Shift the snake inward if its first move would hit a wall."""
        playable_min = 1
        playable_max = self.grid_size - 2

        head_col = self.x[0] // BLOCK_SIZE
        head_row = self.y[0] // BLOCK_SIZE

        dx = 0
        dy = 0

        if self.direction == "LEFT" and head_col == playable_min:
            dx = BLOCK_SIZE
        elif self.direction == "RIGHT" and head_col == playable_max:
            dx = -BLOCK_SIZE
        elif self.direction == "UP" and head_row == playable_min:
            dy = BLOCK_SIZE
        elif self.direction == "DOWN" and head_row == playable_max:
            dy = -BLOCK_SIZE

        if dx != 0 or dy != 0:
            for i in range(self.length):
                self.x[i] += dx
                self.y[i] += dy

    def draw(self):
        """Draw the full map and the snake on the screen."""
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

    def decrease(self):
        """Decrease the snake length by one segment, if possible."""
        if self.length > 0:
            self.length -= 1
            if self.x:
                self.x.pop()
            if self.y:
                self.y.pop()

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
