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
# Added for settings input
COLOR_INPUT_INACTIVE = pygame.Color('lightskyblue3')
COLOR_INPUT_ACTIVE = pygame.Color('dodgerblue2')
COLOR_INPUT_TEXT = (230, 230, 230)

# --- Game States ---
STATE_MAIN_MENU = 0
STATE_BUILDER = 1
STATE_FLIGHT = 2
STATE_SETTINGS = 3
STATE_MULTIPLAYER_LOBBY = 4 # New state for multiplayer menu
STATE_MULTIPLAYER_BUILDER = 5 # Placeholder if needed
STATE_MULTIPLAYER_FLIGHT = 6 # New state for actual multiplayer flight

# --- Button Class (Minor refinement for clarity) ---
class Button:
    def __init__(self, text, rect, font_size=30, color=GRAY, hover_color=LIGHT_GRAY, text_color=BLACK, enabled=True): # Added enabled flag
        # Initialize font if not already done (good practice)
        if not pygame.font.get_init():
            pygame.font.init()
            print("Pygame font initialized.")

        self.rect = pygame.Rect(rect)
        self.text = text
        self.enabled = enabled # Control if button is clickable/changes color
        try:
             self.font = pygame.font.SysFont(None, font_size)
        except Exception as e:
             print(f"Error loading SysFont: {e}. Using default.")
             self.font = pygame.font.Font(None, font_size) # Fallback to default font

        self.base_color = color
        self.hover_color = hover_color
        self.disabled_color = (100, 100, 100) # Color when disabled
        self.text_color = text_color
        self.is_hovered = False

    def draw(self, surface):
        # Determine background color based on state
        if not self.enabled:
             draw_color = self.disabled_color
        elif self.is_hovered:
             draw_color = self.hover_color
        else:
             draw_color = self.base_color

        # Draw the button rectangle
        pygame.draw.rect(surface, draw_color, self.rect, border_radius=5) # Added slight rounding

        # Draw the border (optional, can be removed)
        border_color = (50,50,50) if not self.enabled else self.text_color
        pygame.draw.rect(surface, border_color, self.rect, 2, border_radius=5) # Border matches text color or dark when disabled

        # Render and position the text
        if self.text: # Only render if text exists
            text_surf = self.font.render(self.text, True, self.text_color if self.enabled else GRAY) # Gray text when disabled
            text_rect = text_surf.get_rect(center=self.rect.center)
            surface.blit(text_surf, text_rect)

    def check_hover(self, mouse_pos):
        """ Updates the hover state based on mouse position, only if enabled. """
        if self.enabled:
            self.is_hovered = self.rect.collidepoint(mouse_pos)
        else:
            self.is_hovered = False

    def is_clicked(self, event):
        """ Checks if the button was clicked in this event frame, only if enabled. """
        if not self.enabled:
             return False # Cannot click disabled button
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Check collision with the exact position of the mouse click event
            return self.rect.collidepoint(event.pos)
        return False

# --- InputBox Class (Moved to settings.py) ---
# --- Keep constants used by flight_sim ---
COLOR_SKY_BLUE, COLOR_SPACE_BLACK, COLOR_HORIZON, COLOR_GROUND = (135, 206, 250), (0,0,0), (170, 210, 230), (0, 150, 0)
COLOR_FLAME, COLOR_UI_BAR, COLOR_UI_BAR_BG, COLOR_EXPLOSION = (255,100,0), (0,200,0), (50,50,50), [(255,255,0),(255,150,0),(200,50,0),(150,150,150)]
COLOR_ENGINE_ENABLED, COLOR_ENGINE_DISABLED, COLOR_ACTIVATABLE_READY, COLOR_ACTIVATABLE_USED = GREEN, RED, BLUE, GRAY