import pygame
from snake import Snake
from apple import Apple
from pygame import KEYDOWN, K_UP, K_DOWN, K_LEFT, K_RIGHT, K_RETURN, K_ESCAPE


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

        pygame.init()
        pygame.display.set_caption("Snake Game")
        self.SCREEN_UPDATE = pygame.USEREVENT
        pygame.time.set_timer(self.SCREEN_UPDATE, 400)
        self.surface = pygame.display.set_mode(screen_size)
        # Snake and apple receive the total grid size (including walls)
        self.snake = Snake(self.surface, self.grid_size)
        self.green_apples, self.red_apple = Apple.spawn_apples(
            self.surface,
            self.grid_size,
            self.snake,
        )
        self.snake.draw()

    def game_start_text(self):
        """Display a start message before the game begins."""
        font = pygame.font.SysFont(None, 48)
        text = "Press an arrow key to start"
        color = (255, 255, 255)
        shadow_color = (0, 0, 0)

        text_surface = font.render(text, True, color)
        shadow_surface = font.render(text, True, shadow_color)

        rect = text_surface.get_rect(
            center=(self.surface.get_width() // 2, self.surface.get_height() // 2),
        )

        # Draw a simple shadow for better readability
        shadow_rect = rect.move(2, 2)
        self.surface.blit(shadow_surface, shadow_rect)
        self.surface.blit(text_surface, rect)

    def game_over_text(self):
        """Display a game over message when the player loses."""
        font = pygame.font.SysFont(None, 48)
        text = "You lose! Press Esc to quit"
        color = (255, 255, 255)
        shadow_color = (0, 0, 0)

        text_surface = font.render(text, True, color)
        shadow_surface = font.render(text, True, shadow_color)

        rect = text_surface.get_rect(
            center=(self.surface.get_width() // 2, self.surface.get_height() // 2),
        )

        # Draw a simple shadow for better readability
        shadow_rect = rect.move(2, 2)
        self.surface.blit(shadow_surface, shadow_rect)
        self.surface.blit(text_surface, rect)

    def play(self):
        """Move the snake and draw the apple."""
        self.snake.move()
        for apple in self.green_apples:
            apple.draw()
        self.red_apple.draw()
        self._check_wall_collision()

    def _check_wall_collision(self):
        """Set game_over when the snake's head hits a wall."""
        head_col = self.snake.x[0] // BLOCK_SIZE
        head_row = self.snake.y[0] // BLOCK_SIZE

        if (
            head_col == 0
            or head_col == self.grid_size - 1
            or head_row == 0
            or head_row == self.grid_size - 1
        ):
            self.game_over = True

    def run(self):
        """Run the game."""
        running = True
        started = False
        while running:
            # Handle events
            for event in pygame.event.get():

                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE and self.game_over:
                        # After losing, Esc closes the game.
                        running = False
                    if not self.game_over:
                        if event.key == K_UP:
                            started = True
                            self.snake.move_up()
                        if event.key == K_DOWN:
                            started = True
                            self.snake.move_down()
                        if event.key == K_LEFT:
                            started = True
                            self.snake.move_left()
                        if event.key == K_RIGHT:
                            started = True
                            self.snake.move_right()

                if event.type == pygame.QUIT:
                    running = False
                elif event.type == self.SCREEN_UPDATE:
                    if started and not self.game_over:
                        self.play()
            # If the game hasn't started yet, overlay the start text.
            if not started and not self.game_over:
                self.game_start_text()
            # When game is over, show the game over message.
            if self.game_over:
                self.game_over_text()
            # Update the display
            pygame.display.update()
