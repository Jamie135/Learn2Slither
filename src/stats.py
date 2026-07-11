import math
import pygame

WIDTH, HEIGHT = 900, 660

BG = (13, 15, 22)
PANEL = (20, 24, 34)
BORDER = (44, 52, 70)
TEXT = (230, 236, 245)
TEXT_DIM = (140, 150, 168)
ACCENT = (0, 229, 255)
BAR_SCORE = (57, 255, 122)
BAR_WALL = (255, 64, 80)
BAR_SELF = (255, 170, 60)
BAR_LENGTH = (190, 120, 255)

# Disjoint score buckets (lower bound inclusive) with their slice colors
SCORE_BUCKETS = [
    ("< 10", 0, (108, 117, 140)),
    ("10 - 14", 10, (0, 229, 255)),
    ("15 - 19", 15, (57, 255, 122)),
    ("20 - 24", 20, (255, 214, 64)),
    ("25 - 29", 25, (255, 170, 60)),
    ("30 - 34", 30, (255, 64, 80)),
    (">= 35", 35, (190, 120, 255)),
]


def _font(size, bold=False):
    return pygame.font.SysFont("dejavusansmono", size, bold=bold)


def _text(surface, string, size, color, pos, bold=False,
          center=False, right=False):
    surf = _font(size, bold).render(string, True, color)
    rect = surf.get_rect()
    if center:
        rect.center = pos
    elif right:
        rect.topright = pos
    else:
        rect.topleft = pos
    surface.blit(surf, rect)
    return rect


def _panel(surface, rect, title):
    pygame.draw.rect(surface, PANEL, rect, border_radius=10)
    pygame.draw.rect(surface, BORDER, rect, width=1, border_radius=10)
    _text(surface, title, 20, ACCENT, (rect.left + 16, rect.top + 12),
          bold=True)


def _bar_row(surface, label, value, total, color, x, y, width,
             value_text=None):
    """One horizontal bar with its label and value."""
    ratio = value / total if total > 0 else 0.0
    _text(surface, label, 17, TEXT_DIM, (x, y))
    bar_rect = pygame.Rect(x + 150, y, width - 230, 18)
    pygame.draw.rect(surface, (28, 33, 46), bar_rect, border_radius=6)
    fill = bar_rect.copy()
    fill.width = max(0, int(bar_rect.width * ratio))
    if fill.width > 0:
        pygame.draw.rect(surface, color, fill, border_radius=6)
    if value_text is None:
        value_text = f"{ratio * 100:5.1f} %"
    _text(surface, value_text, 17, TEXT,
          (bar_rect.right + 78, y), right=True)


def _bucket_counts(scores):
    """Count how many sessions fall in each disjoint score bucket."""
    counts = [0] * len(SCORE_BUCKETS)
    for score in scores:
        for i in range(len(SCORE_BUCKETS) - 1, -1, -1):
            if score >= SCORE_BUCKETS[i][1]:
                counts[i] += 1
                break
    return counts


def _pie_slice(surface, center, radius, start_frac, end_frac, color):
    """Draw one pie slice as a polygon fan; fractions are in [0, 1]."""
    # Start at 12 o'clock and go clockwise.
    start = 2 * math.pi * start_frac - math.pi / 2
    end = 2 * math.pi * end_frac - math.pi / 2
    steps = max(2, int((end - start) * radius / 4))
    points = [center]
    for i in range(steps + 1):
        angle = start + (end - start) * i / steps
        points.append((
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
        ))
    pygame.draw.polygon(surface, color, points)


def _draw_score_pie(surface, panel, scores):
    """Pie chart of the score distribution, with a legend on the right."""
    counts = _bucket_counts(scores)
    total = len(scores)
    center = (panel.left + 190, panel.centery + 20)
    radius = 100

    if total == 0:
        pygame.draw.circle(surface, (28, 33, 46), center, radius)
        _text(surface, "no data", 17, TEXT_DIM, center, center=True)
    else:
        frac = 0.0
        for count, (_, _, color) in zip(counts, SCORE_BUCKETS):
            if count == 0:
                continue
            next_frac = frac + count / total
            _pie_slice(surface, center, radius, frac, next_frac, color)
            frac = next_frac
        pygame.draw.circle(surface, BORDER, center, radius, width=1)

    legend_x = panel.left + 380
    y = panel.top + 52
    for count, (label, _, color) in zip(counts, SCORE_BUCKETS):
        pygame.draw.rect(
            surface, color, (legend_x, y + 2, 14, 14), border_radius=3
        )
        pct = count / total * 100 if total > 0 else 0.0
        _text(surface, label, 17, TEXT_DIM, (legend_x + 24, y))
        _text(surface, f"{pct:5.1f} %", 17, TEXT, (legend_x + 150, y))
        _text(surface, f"({count})", 17, TEXT_DIM, (legend_x + 250, y))
        y += 26


def show_stats(mode, sessions, scores, deaths):
    """
    Display the end-of-run statistics screen.

    mode: "Training" or "Evaluation"
    sessions: number of sessions run
    scores: list of per-session max scores
    deaths: dict with counts for "wall", "self" and "length"
    """
    # In headless runs pygame was never initialised; init is a no-op
    # if it already was.
    pygame.init()
    surface = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Snake - {mode} statistics")

    max_score = max(scores) if scores else 0
    avg_score = sum(scores) / len(scores) if scores else 0.0
    total_deaths = sum(deaths.values())

    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key in (
                pygame.K_ESCAPE, pygame.K_RETURN
            ):
                pygame.quit()
                return

        surface.fill(BG)
        _text(surface, f"{mode} statistics", 34, TEXT,
              (WIDTH // 2, 40), bold=True, center=True)
        _text(surface, f"{sessions} session{'s' if sessions > 1 else ''}",
              20, TEXT_DIM, (WIDTH // 2, 74), center=True)

        # Summary panel
        summary = pygame.Rect(40, 105, WIDTH - 80, 90)
        _panel(surface, summary, "Summary")
        _text(surface, f"Max score: {max_score}", 22, BAR_SCORE,
              (summary.left + 16, summary.top + 48), bold=True)
        _text(surface, f"Average score: {avg_score:.2f}", 22, TEXT,
              (summary.centerx, summary.top + 48))

        # Score distribution pie chart
        score_panel = pygame.Rect(40, 215, WIDTH - 80, 255)
        _panel(surface, score_panel, "Score distribution")
        _draw_score_pie(surface, score_panel, scores)

        # Death causes panel
        death_panel = pygame.Rect(40, 490, WIDTH - 80, 130)
        _panel(surface, death_panel, "Causes of death")
        y = death_panel.top + 46
        for label, key, color in (
            ("Wall hit", "wall", BAR_WALL),
            ("Self hit", "self", BAR_SELF),
            ("Length 0", "length", BAR_LENGTH),
        ):
            count = deaths.get(key, 0)
            _bar_row(
                surface, label, count, total_deaths, color,
                death_panel.left + 16, y, death_panel.width - 32,
                value_text=f"{count}",
            )
            y += 28

        _text(surface, "Press Esc or Enter to close", 16, TEXT_DIM,
              (WIDTH // 2, HEIGHT - 18), center=True)

        pygame.display.update()
        clock.tick(30)
