# main.py
import pygame
import sys
import flight_sim
import builder
import os
# Import from the new common file
from ui_elements import Button, STATE_MAIN_MENU, STATE_BUILDER, STATE_FLIGHT, STATE_SETTINGS, SCREEN_WIDTH, SCREEN_HEIGHT, BLACK, WHITE

# --- Main Menu Function ---
def main_menu(screen, clock):
    # Button class is now imported
    buttons = [
        Button("Editor / Build", (SCREEN_WIDTH//2 - 150, 180, 300, 50)),
        Button("Multiplayer (NYI)", (SCREEN_WIDTH//2 - 150, 250, 300, 50)),
        Button("Settings (NYI)", (SCREEN_WIDTH//2 - 150, 320, 300, 50)),
        Button("Quit", (SCREEN_WIDTH//2 - 150, 390, 300, 50)),
    ]
    # Check if a default blueprint exists for launch testing
    default_blueprint = os.path.join("assets", "current_build.json") # Use the builder's autosave
    can_launch_immediately = os.path.exists(default_blueprint)

    if can_launch_immediately:
         # Add a "Quick Launch" button if possible
         buttons.insert(1, Button("Quick Launch Last Build", (SCREEN_WIDTH//2 - 150, 110, 300, 50), color=(100, 180, 100), hover_color=(130, 210, 130)))

    title_font = pygame.font.SysFont(None, 72)
    title_surf = title_font.render("PySpaceFlight", True, WHITE)
    title_rect = title_surf.get_rect(center=(SCREEN_WIDTH//2, 60)) # Adjusted title pos

    while True:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "QUIT"

            button_index_offset = 0
            if can_launch_immediately: button_index_offset = 1 # Account for inserted button

            for i, button in enumerate(buttons):
                if button.is_clicked(event):
                    if i == 0: return STATE_BUILDER # Editor / Build
                    elif can_launch_immediately and i == 1: # Quick Launch
                        return ("LAUNCH", default_blueprint)
                    # Adjust other indices based on offset
                    elif i == (1 + button_index_offset): print("Multiplayer NYI") # Multiplayer
                    elif i == (2 + button_index_offset): print("Settings NYI")    # Settings
                    elif i == (3 + button_index_offset): return "QUIT"             # Quit
                    else: print(f"Button '{button.text}' NYI") # Should not happen

        for button in buttons:
            button.check_hover(mouse_pos)

        screen.fill(BLACK)
        screen.blit(title_surf, title_rect)
        for button in buttons:
            button.draw(screen)
        pygame.display.flip()
        clock.tick(60)

# --- Main Game Loop ---
def main():
    pygame.init()
    # Ensure 'assets' directory exists for blueprints
    assets_dir = "assets"
    if not os.path.exists(assets_dir):
        try:
            os.makedirs(assets_dir)
            print(f"Created '{assets_dir}' directory for blueprints.")
        except OSError as e:
            print(f"Error creating directory {assets_dir}: {e}")
            # Optionally quit if directory cannot be created
            # sys.exit(1)

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("PySpaceFlight - Main Menu")
    clock = pygame.time.Clock()

    current_state = STATE_MAIN_MENU
    blueprint_to_launch = None
    running = True

    while running:
        if current_state == STATE_MAIN_MENU:
            pygame.display.set_caption("PySpaceFlight - Main Menu")
            result = main_menu(screen, clock)
            if result == "QUIT":
                 running = False
            elif isinstance(result, tuple) and result[0] == "LAUNCH": # Handle quick launch from menu
                 blueprint_to_launch = result[1]
                 if blueprint_to_launch and os.path.exists(blueprint_to_launch):
                     current_state = STATE_FLIGHT
                 else:
                     print(f"Error: Quick Launch blueprint '{blueprint_to_launch}' not found.")
                     blueprint_to_launch = None
                     current_state = STATE_MAIN_MENU # Stay in menu
            elif isinstance(result, int): # Standard state change
                 current_state = result
            else: # Should not happen
                 print(f"Unknown main menu result: {result}")
                 current_state = STATE_MAIN_MENU

        elif current_state == STATE_BUILDER:
             pygame.display.set_caption("PySpaceFlight - Rocket Editor")
             builder_result = builder.run_builder(screen, clock) # Call the updated builder

             if isinstance(builder_result, tuple) and builder_result[0] == "LAUNCH":
                 blueprint_to_launch = builder_result[1]
                 if blueprint_to_launch and os.path.exists(blueprint_to_launch):
                     current_state = STATE_FLIGHT
                 else:
                     print(f"Error: Builder returned invalid blueprint '{blueprint_to_launch}' for launch.")
                     blueprint_to_launch = None
                     current_state = STATE_MAIN_MENU # Go back to menu on error
             elif builder_result == STATE_MAIN_MENU:
                 current_state = STATE_MAIN_MENU
             else: # Handle potential quit from builder or other unexpected return
                 print(f"Builder returned unexpected state: {builder_result}. Returning to Main Menu.")
                 current_state = STATE_MAIN_MENU

        elif current_state == STATE_FLIGHT:
             if blueprint_to_launch:
                 pygame.display.set_caption(f"PySpaceFlight - Flying: {os.path.basename(blueprint_to_launch)}")
                 flight_sim.run_simulation(screen, clock, blueprint_to_launch)
                 # Always return to menu after simulation ends (crash, escape, etc.)
                 current_state = STATE_MAIN_MENU
                 blueprint_to_launch = None # Clear blueprint after flight attempt
             else:
                 print("Error: Reached FLIGHT state without a blueprint to launch.")
                 current_state = STATE_MAIN_MENU

        elif current_state == STATE_SETTINGS:
             pygame.display.set_caption("PySpaceFlight - Settings")
             print("Entering Settings State (NYI)")
             # --- Add Settings Menu Function Call Here ---
             # settings_result = settings_menu(screen, clock)
             # current_state = settings_result
             current_state = STATE_MAIN_MENU # Go back to menu for now

        else:
            print(f"Error: Unknown game state: {current_state}")
            running = False # Exit on unknown state

    pygame.quit()
    print("PySpaceFlight Exited.")
    sys.exit()

if __name__ == '__main__':
    main()