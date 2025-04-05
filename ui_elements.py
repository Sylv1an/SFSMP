# ui_elements.py
import pygame

# --- Screen Dimensions ---
SCREEN_WIDTH = 1024 # Increased width slightly
SCREEN_HEIGHT = 768 # Increased height slightly

# --- Colors ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (150, 150, 150)
LIGHT_GRAY = (200, 200, 200)
BLUE = (100, 100, 255)
RED = (255, 100, 100)
GREEN = (100, 255, 100)

# --- Game States ---
STATE_MAIN_MENU = 0
STATE_BUILDER = 1
STATE_FLIGHT = 2
STATE_SETTINGS = 3 # Kept for future use

# --- Button Class (Minor refinement for clarity) ---
class Button:
    def __init__(self, text, rect, font_size=30, color=GRAY, hover_color=LIGHT_GRAY, text_color=BLACK):
        # Initialize font if not already done (good practice)
        if not pygame.font.get_init():
            pygame.font.init()
            print("Pygame font initialized.")

        self.rect = pygame.Rect(rect)
        self.text = text
        try:
             self.font = pygame.font.SysFont(None, font_size)
        except Exception as e:
             print(f"Error loading SysFont: {e}. Using default.")
             self.font = pygame.font.Font(None, font_size) # Fallback to default font

        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.is_hovered = False

    def draw(self, surface):
        # Determine background color based on hover state
        draw_color = self.hover_color if self.is_hovered else self.color

        # Draw the button rectangle
        pygame.draw.rect(surface, draw_color, self.rect, border_radius=5) # Added slight rounding

        # Draw the border (optional, can be removed)
        pygame.draw.rect(surface, self.text_color, self.rect, 2, border_radius=5) # Border matches text color

        # Render and position the text
        if self.text: # Only render if text exists
            text_surf = self.font.render(self.text, True, self.text_color)
            text_rect = text_surf.get_rect(center=self.rect.center)
            surface.blit(text_surf, text_rect)

    def check_hover(self, mouse_pos):
        """ Updates the hover state based on mouse position. """
        self.is_hovered = self.rect.collidepoint(mouse_pos)

    def is_clicked(self, event):
        """ Checks if the button was clicked in this event frame. """
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Check collision with the exact position of the mouse click event
            return self.rect.collidepoint(event.pos)
        return False