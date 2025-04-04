# main.py
import pygame
import sys
import flight_sim
import builder
import os
# Import from the new common file
from ui_elements import Button, STATE_MAIN_MENU, STATE_BUILDER, STATE_FLIGHT, STATE_SETTINGS, SCREEN_WIDTH, SCREEN_HEIGHT, BLACK, WHITE

# --- Constants ---
# Keep screen dimensions here or move to ui_elements if preferred
# SCREEN_WIDTH = 800
# SCREEN_HEIGHT = 600
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)
# GRAY = (150, 150, 150) # Keep colors used only here, or move all to ui_elements
# LIGHT_GRAY = (200, 200, 200)

# --- Game States ---
# Definitions moved to ui_elements.py

# --- Button Class ---
# Definition moved to ui_elements.py

# --- Main Menu Function ---
def main_menu(screen, clock):
    # Button class is now imported
    buttons = [
        Button("Editor / Build", (SCREEN_WIDTH//2 - 150, 180, 300, 50)),
        Button("Multiplayer (NYI)", (SCREEN_WIDTH//2 - 150, 250, 300, 50)),
        Button("Settings (NYI)", (SCREEN_WIDTH//2 - 150, 320, 300, 50)),
        Button("Quit", (SCREEN_WIDTH//2 - 150, 390, 300, 50)),
    ]

    # Rest of main_menu function remains the same...
    while True:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "QUIT"
            for i, button in enumerate(buttons):
                if button.is_clicked(event):
                    if i == 0: return STATE_BUILDER
                    # elif i == 1: print("Multiplayer NYI")
                    # elif i == 2: print("Settings NYI")
                    elif i == 3: return "QUIT"
                    else: print(f"Button '{button.text}' NYI")

        for button in buttons:
            button.check_hover(mouse_pos)

        screen.fill(BLACK)
        title_font = pygame.font.SysFont(None, 72)
        title_surf = title_font.render("PySpaceFlight", True, WHITE)
        title_rect = title_surf.get_rect(center=(SCREEN_WIDTH//2, 80))
        screen.blit(title_surf, title_rect)
        for button in buttons:
            button.draw(screen)
        pygame.display.flip()
        clock.tick(60)


# --- Main Game Loop ---
def main():
    pygame.init()
    assets_dir = "assets"
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
        print(f"Created '{assets_dir}' directory.")

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("PySpaceFlight - Main Menu")
    clock = pygame.time.Clock()

    current_state = STATE_MAIN_MENU # Use imported constant
    blueprint_to_launch = None
    running = True

    while running:
        if current_state == STATE_MAIN_MENU:
            pygame.display.set_caption("PySpaceFlight - Main Menu")
            result = main_menu(screen, clock)
            if result == "QUIT": running = False
            else: current_state = result

        elif current_state == STATE_BUILDER: # Use imported constant
             pygame.display.set_caption("PySpaceFlight - Rocket Editor")
             builder_result = builder.run_builder(screen, clock) # Call builder

             if isinstance(builder_result, tuple) and builder_result[0] == "LAUNCH":
                 blueprint_to_launch = builder_result[1]
                 if blueprint_to_launch and os.path.exists(blueprint_to_launch):
                     current_state = STATE_FLIGHT # Use imported constant
                 else:
                     print(f"Error: Blueprint file '{blueprint_to_launch}' not found.")
                     blueprint_to_launch = None
                     current_state = STATE_MAIN_MENU # Use imported constant
             elif builder_result == STATE_MAIN_MENU: # Use imported constant
                 current_state = STATE_MAIN_MENU
             else: # Handle quit from builder or unexpected return
                 current_state = STATE_MAIN_MENU

        elif current_state == STATE_FLIGHT: # Use imported constant
             if blueprint_to_launch:
                 pygame.display.set_caption(f"PySpaceFlight - Flying: {os.path.basename(blueprint_to_launch)}")
                 flight_sim.run_simulation(screen, clock, blueprint_to_launch)
                 current_state = STATE_MAIN_MENU # Return to menu after flight
                 blueprint_to_launch = None
             else:
                 print("Error: No blueprint for flight.")
                 current_state = STATE_MAIN_MENU

        elif current_state == STATE_SETTINGS: # Use imported constant
             pygame.display.set_caption("PySpaceFlight - Settings")
             print("Entering Settings State (NYI)")
             current_state = STATE_MAIN_MENU

        else:
            print(f"Unknown game state: {current_state}")
            running = False

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()