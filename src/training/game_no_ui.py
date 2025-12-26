import numpy as np
from snake_no_ui import SnakeTraining
from apple_no_ui import AppleTraining


BLOCK_SIZE = 40


class GameTraining:
    def __init__(self, grid_size):
        """Initialise the game for training (no UI)."""
        self.playable_grid_size = grid_size
        # Total grid, including the surrounding wall layer
        self.grid_size = grid_size + 2
        self.game_over = False
        # Snake and apple receive the total grid size (including walls)
        self.snake = SnakeTraining(self.grid_size)
        self.green_apples, self.red_apple = AppleTraining.spawn_apples(
            self.grid_size,
            self.snake,
        )
        self.reward = 0

    def play(self):
        """Move the snake and handle game logic."""
        self.snake.move()
        self.reward = -0.1
        self.eats_apple()
        # If eating a red apple reduced the length to 0, the game is over
        if self.game_over or self.snake.length == 0:
            return
        self.check_wall_collision()
        self.check_self_collision()

    def eats_apple(self):
        """Handle collisions between the snake's head and apples."""
        head_pos = (self.snake.x[0], self.snake.y[0])

        # Green apples: increase length and respawn the eaten apple.
        for apple in self.green_apples:
            if (apple.x, apple.y) == head_pos:
                self.snake.increase()
                # Build occupied positions: snake + all other apples.
                occupied = {
                    (self.snake.x[i], self.snake.y[i])
                    for i in range(self.snake.length)
                }
                for other in self.green_apples:
                    if other is not apple:
                        occupied.add((other.x, other.y))
                occupied.add((self.red_apple.x, self.red_apple.y))
                apple.randomize_position(occupied)
                self.reward = 10

        # Red apple: decrease length and respawn, or end game if length is 0.
        if (self.red_apple.x, self.red_apple.y) == head_pos:
            self.snake.decrease()
            if self.snake.length == 0:
                self.game_over = True
                return

            occupied = {
                (self.snake.x[i], self.snake.y[i])
                for i in range(self.snake.length)
            }
            for apple in self.green_apples:
                occupied.add((apple.x, apple.y))
            self.red_apple.randomize_position(occupied)
            self.reward = -10

    def check_wall_collision(self):
        """Set game_over when the snake's head hits a wall."""
        if self.snake.length == 0:
            return
        head_col = self.snake.x[0] // BLOCK_SIZE
        head_row = self.snake.y[0] // BLOCK_SIZE

        if (
            head_col == 0
            or head_col == self.grid_size - 1
            or head_row == 0
            or head_row == self.grid_size - 1
        ):
            self.game_over = True
            self.reward = -100

    def check_self_collision(self):
        """Set game_over when the snake's head collides with its own body."""
        if self.snake.length == 0:
            return
        head_pos = (self.snake.x[0], self.snake.y[0])
        for i in range(1, self.snake.length):
            if (self.snake.x[i], self.snake.y[i]) == head_pos:
                self.game_over = True
                self.reward = -100
                break

    def reset(self):
        """Reset the game to the initial state."""
        self.game_over = False
        self.snake = SnakeTraining(self.grid_size)
        self.green_apples, self.red_apple = AppleTraining.spawn_apples(
            self.grid_size,
            self.snake,
        )

    def next_direction(self, action):
        """Set the snake's direction based on the action input."""
        if np.array_equal(action, [1, 0, 0, 0]):
            new_dir = "right"
        elif np.array_equal(action, [0, 1, 0, 0]):
            new_dir = "down"
        elif np.array_equal(action, [0, 0, 1, 0]):
            new_dir = "left"
        elif np.array_equal(action, [0, 0, 0, 1]):
            new_dir = "up"
        return new_dir

    def set_direction(self, direction):
        """Set the snake's direction based on input (for training control)."""
        if direction == "LEFT":
            self.snake.move_left()
        elif direction == "RIGHT":
            self.snake.move_right()
        elif direction == "UP":
            self.snake.move_up()
        elif direction == "DOWN":
            self.snake.move_down()

    def run(self, action):
        """Run the game."""

        direction = self.next_direction(action)
        if direction == "left":
            self.snake.move_left()
        elif direction == "right":
            self.snake.move_right()
        elif direction == "up":
            self.snake.move_up()
        elif direction == "down":
            self.snake.move_down()

        while not self.game_over:
            self.play()

        return self.reward, self.game_over, self.snake.length - 3
