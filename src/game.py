import numpy as np
from snake import Snake
from apple import Apple

try:
    import pygame
    from pygame import (
        KEYDOWN,
        K_ESCAPE,
        K_RETURN,
        K_1,
        K_2,
    )
except ImportError:
    pygame = None


BLOCK_SIZE = 40


class Game:
    def __init__(self, grid_size, no_ui=False, show_state=False):
        """Initialise the game."""
        self.playable_grid_size = grid_size
        # Total grid, including the surrounding wall layer
        self.grid_size = grid_size + 2
        self.game_over = False
        self.no_ui = no_ui
        self.show_state = show_state
        self.reward = 0
        # Why the last game ended: "wall", "self" or "length"
        self.death_cause = None

        if not self.no_ui:
            screen_size = (
                self.grid_size * BLOCK_SIZE,
                self.grid_size * BLOCK_SIZE,
            )
            self.timer = 400
            pygame.init()
            pygame.display.set_caption("Snake Game")
            self.SCREEN_UPDATE = pygame.USEREVENT
            pygame.time.set_timer(self.SCREEN_UPDATE, self.timer)
            self.surface = pygame.display.set_mode(screen_size)
        else:
            self.surface = None

        # Snake and apple receive the total grid size (including walls)
        self.snake = Snake(self.surface, self.grid_size)
        self.green_apples, self.red_apple = Apple.spawn_apples(
            self.surface,
            self.grid_size,
            self.snake,
        )
        if not self.no_ui:
            self.snake.draw()

    def draw_snake_length(self):
        """Display the current snake length on the screen."""
        font = pygame.font.SysFont(None, 30)
        text = f"Score: {self.snake.length - 3}"
        color = (255, 255, 255)
        shadow_color = (0, 0, 0)

        text_surface = font.render(text, True, color)
        shadow_surface = font.render(text, True, shadow_color)

        # Position near the top-left corner, with a small margin.
        rect = text_surface.get_rect(topleft=(10, 10))
        shadow_rect = rect.move(2, 2)

        self.surface.blit(shadow_surface, shadow_rect)
        self.surface.blit(text_surface, rect)

    def is_danger(self, point):
        """Check if the next position is dangerous."""
        point_x = point[0]
        point_y = point[1]
        for i in range(1, self.snake.length):
            if self.snake.x[i] == point_x and self.snake.y[i] == point_y:
                return True
        if (
            point_x <= 0 or point_x >= (self.grid_size - 1) * BLOCK_SIZE
            or point_y <= 0 or point_y >= (self.grid_size - 1) * BLOCK_SIZE
        ):
            return True
        return False

    def game_start_text(self):
        """Display a start message before the game begins."""
        font = pygame.font.SysFont(None, 48)
        text = "Press Enter to start"
        color = (255, 255, 255)
        shadow_color = (0, 0, 0)

        text_surface = font.render(text, True, color)
        shadow_surface = font.render(text, True, shadow_color)

        rect = text_surface.get_rect(
            center=(
                self.surface.get_width() // 2,
                self.surface.get_height() // 2,
            ),
        )

        # Draw a simple shadow for better readability
        shadow_rect = rect.move(2, 2)
        self.surface.blit(shadow_surface, shadow_rect)
        self.surface.blit(text_surface, rect)

    def game_over_text(self):
        """Display a game over message when the player loses."""
        font = pygame.font.SysFont(None, 24)
        text = "You lose! Press Esc to quit or Enter to restart"
        color = (255, 255, 255)
        shadow_color = (0, 0, 0)

        text_surface = font.render(text, True, color)
        shadow_surface = font.render(text, True, shadow_color)

        rect = text_surface.get_rect(
            center=(
                self.surface.get_width() // 2,
                self.surface.get_height() // 2,
            ),
        )

        # Draw a simple shadow for better readability
        shadow_rect = rect.move(2, 2)
        self.surface.blit(shadow_surface, shadow_rect)
        self.surface.blit(text_surface, rect)

    def play(self):
        """Move the snake and handle game logic."""
        self.snake.move()
        self.reward = -0.01
        self.eats_apple()
        # If eating a red apple reduced the length to 0, the game is over;
        # avoid accessing self.snake.x[0] which would raise IndexError.
        if self.snake.length == 0:
            self.game_over = True
            self.death_cause = "length"
            return
        if self.game_over:
            return
        if not self.no_ui:
            for apple in self.green_apples:
                apple.draw()
            self.red_apple.draw()
            self.draw_snake_length()
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
                self.reward = 20

        # Red apple: decrease length and respawn, or end game if length is 0.
        if (self.red_apple.x, self.red_apple.y) == head_pos:
            self.snake.decrease()
            if self.snake.length == 0:
                self.game_over = True
                self.death_cause = "length"
                return

            occupied = {
                (self.snake.x[i], self.snake.y[i])
                for i in range(self.snake.length)
            }
            for apple in self.green_apples:
                occupied.add((apple.x, apple.y))
            self.red_apple.randomize_position(occupied)
            self.reward = -5

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
            self.death_cause = "wall"
            self.reward = -20

    def check_self_collision(self):
        """Set game_over when the snake's head collides with its own body."""
        if self.snake.length == 0:
            return
        head_pos = (self.snake.x[0], self.snake.y[0])
        for i in range(1, self.snake.length):
            if (self.snake.x[i], self.snake.y[i]) == head_pos:
                self.game_over = True
                self.death_cause = "self"
                self.reward = -20
                break

    def reset(self):
        """Reset the game to the initial state."""
        self.game_over = False
        self.reward = 0
        self.death_cause = None
        self.snake = Snake(self.surface, self.grid_size)
        self.green_apples, self.red_apple = Apple.spawn_apples(
            self.surface,
            self.grid_size,
            self.snake,
        )
        if not self.no_ui:
            self.snake.draw()

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
        else:
            raise ValueError(f"Invalid action: {action}")
        return new_dir

    def _apply_direction(self, action):
        """Apply direction from action array to the snake."""
        direction = self.next_direction(action)
        if direction == "right":
            self.snake.move_right()
        elif direction == "left":
            self.snake.move_left()
        elif direction == "up":
            self.snake.move_up()
        elif direction == "down":
            self.snake.move_down()
        return direction

    def _cell_symbol(self, cell_col, cell_row):
        """Return the symbol representing a board cell."""
        if (
            cell_col <= 0
            or cell_col >= self.grid_size - 1
            or cell_row <= 0
            or cell_row >= self.grid_size - 1
        ):
            return "W"

        if self.snake.length > 0:
            head_col = self.snake.x[0] // BLOCK_SIZE
            head_row = self.snake.y[0] // BLOCK_SIZE
            if (cell_col, cell_row) == (head_col, head_row):
                return "H"

            for i in range(1, self.snake.length):
                body_col = self.snake.x[i] // BLOCK_SIZE
                body_row = self.snake.y[i] // BLOCK_SIZE
                if (cell_col, cell_row) == (body_col, body_row):
                    return "S"

        for apple in self.green_apples:
            apple_col = apple.x // BLOCK_SIZE
            apple_row = apple.y // BLOCK_SIZE
            if (cell_col, cell_row) == (apple_col, apple_row):
                return "G"

        red_col = self.red_apple.x // BLOCK_SIZE
        red_row = self.red_apple.y // BLOCK_SIZE
        if (cell_col, cell_row) == (red_col, red_row):
            return "R"

        return "0"

    def _print_step_state(self, requested_direction, effective_direction):
        """Print the requested action, effective direction, and local vision."""
        print(f"Action taken: {effective_direction}")

        if self.snake.length == 0:
            print("Snake vision: unavailable (snake has no segments).")
            print()
            return

        head_col = self.snake.x[0] // BLOCK_SIZE
        head_row = self.snake.y[0] // BLOCK_SIZE
        print("Snake vision:")
        for row in range(self.grid_size):
            row_cells = []
            for col in range(self.grid_size):
                if col == head_col or row == head_row:
                    row_cells.append(self._cell_symbol(col, row))
                else:
                    row_cells.append(" ")
            print(" ".join(row_cells))
        print()

    def play_step(self, action, step=False):
        """
        Execute one game step immediately (for AI control).
        Always steps regardless of UI mode. Renders if UI is on.
        In step-by-step mode, waits for a key press before returning.
        Returns: (reward, game_over, score)
        """
        requested_direction = self._apply_direction(action)
        self.play()

        if not self.no_ui and self.show_state:
            self._print_step_state(
                requested_direction,
                self.snake.direction.lower(),
            )

        if not self.no_ui:
            # Process events (quit, speed control)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        pygame.quit()
                        quit()
                    elif event.key == K_2:
                        self.timer -= 50
                        if self.timer < 50:
                            self.timer = 50
                    elif event.key == K_1:
                        self.timer += 50
                        if self.timer > 800:
                            self.timer = 800

            if self.game_over:
                self.game_over_text()
            pygame.display.update()

            if step:
                self._wait_for_keypress()
            else:
                # Control speed using timer value as delay
                pygame.time.delay(self.timer)

        return self.reward, self.game_over, self.snake.length - 3

    def _wait_for_keypress(self):
        """Wait for Space, Enter, or arrow key to advance one step."""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        pygame.quit()
                        quit()
                    elif event.key in (
                        pygame.K_SPACE, K_RETURN,
                        pygame.K_RIGHT, pygame.K_LEFT,
                        pygame.K_UP, pygame.K_DOWN,
                    ):
                        return

    def run(self, action):
        """Run one step of the game (legacy event-driven UI mode)."""

        self._apply_direction(action)

        if self.no_ui:
            self.play()
            return self.reward, self.game_over, self.snake.length - 3

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == self.SCREEN_UPDATE:
                if not self.game_over:
                    self.play()
                    pygame.display.update()
                    pygame.time.Clock().tick(200)
                    break
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE and self.game_over:
                    # After losing, Esc closes the game.
                    pygame.quit()
                    quit()
                if event.key == K_RETURN and self.game_over:
                    self.reset()
                if not self.game_over:

                    if event.key == K_2:
                        # Increase speed: shorter interval between updates.
                        self.timer -= 50
                        if self.timer < 50:
                            self.timer = 50
                        pygame.time.set_timer(
                            self.SCREEN_UPDATE,
                            self.timer,
                        )
                    if event.key == K_1:
                        # Decrease speed: longer interval between updates.
                        self.timer += 50
                        if self.timer > 800:
                            self.timer = 800
                        pygame.time.set_timer(
                            self.SCREEN_UPDATE,
                            self.timer,
                        )
        return self.reward, self.game_over, self.snake.length - 3
