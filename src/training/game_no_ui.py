import pygame
from snake import Snake
from apple import Apple
from pygame import (
    KEYDOWN,
    K_ESCAPE,
    K_RETURN,
    K_1,
    K_2,
)


BLOCK_SIZE = 40


class Game:
    def __init__(self, grid_size):
        """Initialise the game."""
        self.playable_grid_size = grid_size
        # Total grid, including the surrounding wall layer
        self.grid_size = grid_size + 2
        self.game_over = False
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
        # Snake and apple receive the total grid size (including walls)
        self.snake = Snake(self.surface, self.grid_size)
        self.green_apples, self.red_apple = Apple.spawn_apples(
            self.surface,
            self.grid_size,
            self.snake,
        )
        self.snake.draw()

    def draw_snake_length(self):
        """Display the current snake length on the screen."""
        font = pygame.font.SysFont(None, 30)
        text = f"Length: {self.snake.length}"
        color = (255, 255, 255)
        shadow_color = (0, 0, 0)

        text_surface = font.render(text, True, color)
        shadow_surface = font.render(text, True, shadow_color)

        # Position near the top-left corner, with a small margin.
        rect = text_surface.get_rect(topleft=(10, 10))
        shadow_rect = rect.move(2, 2)

        self.surface.blit(shadow_surface, shadow_rect)
        self.surface.blit(text_surface, rect)

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
        """Move the snake and draw the apple."""
        self.snake.move()
        self.eats_apple()
        # If eating a red apple reduced the length to 0, the game is over;
        # avoid accessing self.snake.x[0] which would raise IndexError.
        if self.game_over or self.snake.length == 0:
            return
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
                    (self.snake.x[i], self.snake.y[i]) for i in range(self.snake.length)
                }
                for other in self.green_apples:
                    if other is not apple:
                        occupied.add((other.x, other.y))
                occupied.add((self.red_apple.x, self.red_apple.y))
                apple.randomize_position(occupied)

        # Red apple: decrease length and respawn, or end game if length is 0.
        if (self.red_apple.x, self.red_apple.y) == head_pos:
            self.snake.decrease()
            if self.snake.length == 0:
                self.game_over = True
                return

            occupied = {
                (self.snake.x[i], self.snake.y[i]) for i in range(self.snake.length)
            }
            for apple in self.green_apples:
                occupied.add((apple.x, apple.y))
            self.red_apple.randomize_position(occupied)

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

    def check_self_collision(self):
        """Set game_over when the snake's head collides with its own body."""
        if self.snake.length == 0:
            return
        head_pos = (self.snake.x[0], self.snake.y[0])
        for i in range(1, self.snake.length):
            if (self.snake.x[i], self.snake.y[i]) == head_pos:
                self.game_over = True
                break
    
    def reset(self):
        """Reset the game to the initial state."""
        self.game_over = False
        self.snake = Snake(self.surface, self.grid_size)
        self.green_apples, self.red_apple = Apple.spawn_apples(
            self.surface,
            self.grid_size,
            self.snake,
        )
        self.snake.draw()

    def next_direction(self, action):
        """Set the snake's direction based on the action input."""
        pass
        

    def run(self):
        """Run the game."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == self.SCREEN_UPDATE:
                if not self.game_over:
                    self.play()
                    pygame.display.update()
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
