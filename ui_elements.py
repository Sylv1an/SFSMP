# ui_elements.py
import pygame

# --- Screen Dimensions ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# --- Colors ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (150, 150, 150)
LIGHT_GRAY = (200, 200, 200)
BLUE = (100, 100, 255) # For UI feedback, builder
RED = (255, 100, 100)  # For UI feedback, builder
GREEN = (100, 255, 100) # For UI feedback, builder

# --- Game States ---
STATE_MAIN_MENU = 0
STATE_BUILDER = 1
STATE_FLIGHT = 2
STATE_SETTINGS = 3

# --- Button Class (Moved from main.py) ---
class Button:
    def __init__(self, text, rect, font_size=30, color=GRAY, hover_color=LIGHT_GRAY):
        # Initialize font only once if not already done
        if not pygame.font.get_init():
            pygame.font.init()
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = pygame.font.SysFont(None, font_size)
        self.color = color
        self.hover_color = hover_color
        self.is_hovered = False

    def draw(self, surface):
        draw_color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(surface, draw_color, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 2) # Border

        text_surf = self.font.render(self.text, True, BLACK)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def check_hover(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)

    def is_clicked(self, event):
        # Checks event type and if the CLICK POSITION (event.pos) is inside the rect
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(event.pos) # Use event position directly!
        return False