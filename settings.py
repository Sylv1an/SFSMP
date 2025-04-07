# settings.py
import json
import os

SETTINGS_FILE = "settings.json"
DEFAULT_SETTINGS = {
    "host_ip": "127.0.0.1", # Default to localhost
    "port": 65432,          # Default port
    "player_name": "Pilot"  # Default player name
}

settings_data = {}

def load_settings():
    """Loads settings from the JSON file, using defaults if file not found."""
    global settings_data
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                loaded = json.load(f)
                # Merge with defaults to ensure all keys exist
                settings_data = {**DEFAULT_SETTINGS, **loaded}
                print(f"Settings loaded from {SETTINGS_FILE}")
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading settings from {SETTINGS_FILE}: {e}. Using defaults.")
            settings_data = DEFAULT_SETTINGS.copy()
    else:
        print(f"Settings file '{SETTINGS_FILE}' not found. Using defaults.")
        settings_data = DEFAULT_SETTINGS.copy()
        save_settings() # Create the file with defaults

def save_settings():
    """Saves the current settings dictionary to the JSON file."""
    global settings_data
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings_data, f, indent=4)
        # print(f"Settings saved to {SETTINGS_FILE}") # Less verbose saving
    except IOError as e:
        print(f"Error saving settings to {SETTINGS_FILE}: {e}")

def get_setting(key):
    """Gets a specific setting value."""
    global settings_data
    return settings_data.get(key)

def set_setting(key, value):
    """Sets a specific setting value and saves immediately."""
    global settings_data
    settings_data[key] = value
    save_settings()

# --- Simple Input Box for Settings Menu ---
class InputBox:
    def __init__(self, x, y, w, h, text='', font=None, max_len=30):
        self.rect = pygame.Rect(x, y, w, h)
        self.color_inactive = pygame.Color('lightskyblue3')
        self.color_active = pygame.Color('dodgerblue2')
        self.color = self.color_inactive
        self.text = text
        self.font = font if font else pygame.font.Font(None, 28) # Default font size 28
        self.txt_surface = self.font.render(text, True, self.color)
        self.active = False
        self.max_len = max_len
        self._update_text_surface() # Initial rendering

    def _update_text_surface(self):
         # Ensure text color contrasts with background
         text_color = (255,255,255) if self.active else (230,230,230)
         self.txt_surface = self.font.render(self.text, True, text_color)
         # Adjust rect width if text is too long (optional, could clip instead)
         # self.rect.w = max(self.initial_w, self.txt_surface.get_width()+10)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
            self.color = self.color_active if self.active else self.color_inactive
            self._update_text_surface() # Update color on activation change

        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    self.active = False # Deactivate on Enter
                    self.color = self.color_inactive
                    self._update_text_surface()
                    return "ENTER" # Signal Enter key press
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                elif event.unicode.isprintable() and len(self.text) < self.max_len: # Check if printable and length limit
                    self.text += event.unicode
                self._update_text_surface() # Re-render the text.
        return None # No special action signalled

    def update(self):
        # Resize the box if the text is too long (optional - might clip instead)
        # width = max(200, self.txt_surface.get_width()+10)
        # self.rect.w = width
        pass # No dynamic updates needed currently

    def draw(self, screen):
        # Draw the text centered within the box height
        text_y = self.rect.y + (self.rect.height - self.txt_surface.get_height()) // 2
        screen.blit(self.txt_surface, (self.rect.x+5, text_y))
        # Draw the rectangle border
        pygame.draw.rect(screen, self.color, self.rect, 2)


# Load settings when the module is imported
import pygame # Need pygame for InputBox
pygame.font.init() # Ensure font system is ready for InputBox
load_settings()