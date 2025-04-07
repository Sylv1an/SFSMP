# main.py
import pygame
import sys
import flight_sim
import builder
import os
# Import from the new common file
from ui_elements import (Button, # Removed InputBox from here
                        STATE_MAIN_MENU, STATE_BUILDER, STATE_FLIGHT, STATE_SETTINGS,
                        STATE_MULTIPLAYER_LOBBY, STATE_MULTIPLAYER_FLIGHT, # Added MP states
                        SCREEN_WIDTH, SCREEN_HEIGHT, BLACK, WHITE, GRAY, LIGHT_GRAY, RED, GREEN)
# Import settings functions
import settings
# Import network classes
import network

# --- Global Network Variables ---
network_manager = None # Will hold Server or Client instance
multiplayer_mode = None # "HOST" or "CLIENT"
multiplayer_target_ip = None
multiplayer_port = None
player_name = None

# --- Font Initialization ---
pygame.font.init() # Ensure font is initialized early
title_font = pygame.font.SysFont(None, 72)
menu_font = pygame.font.SysFont(None, 36)
input_font = pygame.font.SysFont(None, 28) # Font for input boxes

# --- Main Menu Function ---
def main_menu(screen, clock):
    buttons = [
        Button("Editor / Build", (SCREEN_WIDTH//2 - 150, 180, 300, 50)),
        # Changed Multiplayer button
        Button("Multiplayer", (SCREEN_WIDTH//2 - 150, 250, 300, 50)),
        Button("Settings", (SCREEN_WIDTH//2 - 150, 320, 300, 50)),
        Button("Quit", (SCREEN_WIDTH//2 - 150, 390, 300, 50)),
    ]
    # Check if a default blueprint exists for launch testing
    default_blueprint = os.path.join("assets", "current_build.json") # Use the builder's autosave
    can_launch_immediately = os.path.exists(default_blueprint)

    if can_launch_immediately:
         # Add a "Quick Launch" button if possible
         buttons.insert(1, Button("Quick Launch Last Build", (SCREEN_WIDTH//2 - 150, 110, 300, 50), color=(100, 180, 100), hover_color=(130, 210, 130)))

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
                    elif i == (1 + button_index_offset): return STATE_MULTIPLAYER_LOBBY # Multiplayer Lobby
                    elif i == (2 + button_index_offset): return STATE_SETTINGS         # Settings
                    elif i == (3 + button_index_offset): return "QUIT"                 # Quit
                    else: print(f"Button '{button.text}' NYI") # Should not happen

        for button in buttons:
            button.check_hover(mouse_pos)

        screen.fill(BLACK)
        screen.blit(title_surf, title_rect)
        for button in buttons:
            button.draw(screen)
        pygame.display.flip()
        clock.tick(60)

# --- Settings Menu Function ---
def settings_menu(screen, clock):
    global player_name # Allow modification
    input_box_ip = settings.InputBox(SCREEN_WIDTH // 2 - 150, 150, 300, 40, text=settings.get_setting("host_ip"), font=input_font, max_len=40)
    input_box_port = settings.InputBox(SCREEN_WIDTH // 2 - 150, 230, 140, 40, text=str(settings.get_setting("port")), font=input_font, max_len=5)
    input_box_name = settings.InputBox(SCREEN_WIDTH // 2 - 150, 310, 300, 40, text=settings.get_setting("player_name"), font=input_font, max_len=20)
    input_boxes = [input_box_ip, input_box_port, input_box_name]

    back_button = Button("Back", (SCREEN_WIDTH // 2 - 75, 400, 150, 50))
    active_box = None

    while True:
        mouse_pos = pygame.mouse.get_pos()
        # REMOVED result = None from here

        for event in pygame.event.get():
            if event.type == pygame.QUIT: return "QUIT"
            if event.type == pygame.KEYDOWN:
                 if event.key == pygame.K_ESCAPE:
                      # Optional: Save on ESCAPE too if you want
                      # settings.set_setting("host_ip", input_box_ip.text)
                      # ... etc ...
                      return STATE_MAIN_MENU

            # Handle input box events
            current_active = None
            for box in input_boxes:
                 box_result = box.handle_event(event)
                 if box.active: current_active = box
                 if box_result == "ENTER" and active_box == box:
                      active_box.active = False
                      active_box.color = active_box.color_inactive
                      active_box._update_text_surface() # Use internal method to update text surface
                      active_box = None
                      # Consider if break is needed here? Maybe not, allow other events.

            if current_active: active_box = current_active
            # Removed setting active_box to None here, handle_event manages activation

            # Handle back button click
            if back_button.is_clicked(event):
                 # --- FIX: Moved Save and Return inside the click check ---
                 print("Back button clicked, saving settings...")
                 settings.set_setting("host_ip", input_box_ip.text)
                 try:
                     port_val = int(input_box_port.text)
                     if 0 < port_val < 65536:
                         settings.set_setting("port", port_val)
                     else: print("Invalid port number, keeping old value.")
                 except ValueError:
                     print("Invalid port number format, keeping old value.")
                 new_pname = input_box_name.text.strip()
                 # Use default name from settings if input is empty
                 if not new_pname: new_pname = settings.DEFAULT_SETTINGS["player_name"]
                 settings.set_setting("player_name", new_pname)
                 player_name = new_pname # Update global variable if used elsewhere

                 return STATE_MAIN_MENU # Return immediately after saving

        # --- REMOVED the "if result:" block from outside the event loop ---

        # Update UI elements (only runs if no exit event occurred)
        back_button.check_hover(mouse_pos)
        for box in input_boxes: box.update() # InputBox update might be needed for blinking cursor later

        # --- Drawing ---
        screen.fill(BLACK)
        title_surf = title_font.render("Settings", True, WHITE)
        title_rect = title_surf.get_rect(center=(SCREEN_WIDTH//2, 60))
        screen.blit(title_surf, title_rect)

        # Draw labels for input boxes
        label_ip = menu_font.render("Host/Server IP:", True, WHITE)
        screen.blit(label_ip, (input_box_ip.rect.x, input_box_ip.rect.y - 30))
        label_port = menu_font.render("Port:", True, WHITE)
        screen.blit(label_port, (input_box_port.rect.x, input_box_port.rect.y - 30))
        label_name = menu_font.render("Player Name:", True, WHITE)
        screen.blit(label_name, (input_box_name.rect.x, input_box_name.rect.y - 30))

        for box in input_boxes: box.draw(screen)
        back_button.draw(screen)

        pygame.display.flip()
        clock.tick(60)

# --- Multiplayer Lobby ---
def multiplayer_lobby(screen, clock):
    global network_manager, multiplayer_mode, multiplayer_target_ip, multiplayer_port, player_name

    # Ensure network manager is cleaned up if we re-enter lobby
    if network_manager:
        network_manager.stop()
        network_manager = None

    # Get settings
    multiplayer_target_ip = settings.get_setting("host_ip")
    multiplayer_port = settings.get_setting("port")
    player_name = settings.get_setting("player_name")

    host_button = Button("Host Game", (SCREEN_WIDTH//2 - 150, 150, 300, 50))
    join_button = Button("Join Game", (SCREEN_WIDTH//2 - 150, 220, 300, 50))
    back_button = Button("Back to Menu", (SCREEN_WIDTH//2 - 150, 360, 300, 50))
    status_text = "Choose Host or Join"
    status_color = WHITE

    buttons = [host_button, join_button, back_button]

    while True:
        mouse_pos = pygame.mouse.get_pos()
        # Check network queue for errors during connection attempts
        if network_manager:
            try:
                msg = network_manager.message_queue.get_nowait()
                print(f"Lobby Network Msg: {msg}") # Debug
                if msg.get("type") == network.MSG_TYPE_ERROR:
                    status_text = f"Error: {msg.get('data', 'Unknown network error')}"
                    status_color = RED
                    if network_manager: network_manager.stop() # Stop on error
                    network_manager = None
                    multiplayer_mode = None
                elif msg.get("type") == network.MSG_TYPE_CONNECT_OK:
                    status_text = f"Connected! Assigned ID: {msg.get('pid')}. Ready!"
                    status_color = GREEN
                    network_manager.player_id = msg.get('pid') # Store assigned ID
                    network_manager.player_name = msg.get('name') # Store assigned name
                    # Transition to builder or flight state
                    return STATE_BUILDER # Go to builder after successful connection
                # Handle other messages if needed in lobby (e.g., player list)

            except queue.Empty:
                pass # No messages

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return "QUIT"
            if event.type == pygame.KEYDOWN:
                 if event.key == pygame.K_ESCAPE:
                      if network_manager: network_manager.stop()
                      network_manager = None
                      return STATE_MAIN_MENU

            clicked_button = None
            for button in buttons:
                 if button.is_clicked(event):
                      clicked_button = button
                      break

            if clicked_button == host_button:
                 print("Attempting to host...")
                 status_text = f"Starting server on port {multiplayer_port}..."
                 status_color = WHITE
                 pygame.display.flip() # Show status update immediately

                 network_manager = network.Server(port=multiplayer_port)
                 if network_manager.start():
                     status_text = "Server started! Waiting for connections..."
                     status_color = GREEN
                     multiplayer_mode = "HOST"
                     network_manager.player_id = 0 # Host is player 0
                     network_manager.player_name = player_name
                     # Add host to client list (for sending blueprints etc later)
                     # The server doesn't connect to itself, so we manually add
                     # We use a dummy socket object or None
                     network_manager.clients[None] = {'id': 0, 'name': player_name, 'address': 'localhost'}
                     # Host goes directly to builder
                     return STATE_BUILDER
                 else:
                     status_text = "Failed to start server! Check port/IP."
                     status_color = RED
                     network_manager = None

            elif clicked_button == join_button:
                 print(f"Attempting to join {multiplayer_target_ip}:{multiplayer_port} as {player_name}...")
                 status_text = f"Connecting to {multiplayer_target_ip}:{multiplayer_port}..."
                 status_color = WHITE
                 pygame.display.flip() # Show status update immediately

                 network_manager = network.Client()
                 if network_manager.connect(multiplayer_target_ip, multiplayer_port, player_name):
                     # Connection successful, waiting for CONNECT_OK message
                     status_text = "Connected! Waiting for Server OK..."
                     status_color = WHITE
                     multiplayer_mode = "CLIENT"
                     # Wait for CONNECT_OK message in the network queue loop above
                 else:
                     status_text = f"Failed to connect! Check IP/Port/Server."
                     status_color = RED
                     network_manager = None # Clean up failed client


            elif clicked_button == back_button:
                 if network_manager: network_manager.stop() # Ensure cleanup if backing out
                 network_manager = None
                 return STATE_MAIN_MENU


        # Update UI
        for button in buttons: button.check_hover(mouse_pos)

        # Drawing
        screen.fill(BLACK)
        title_surf = title_font.render("Multiplayer Lobby", True, WHITE)
        title_rect = title_surf.get_rect(center=(SCREEN_WIDTH//2, 60))
        screen.blit(title_surf, title_rect)

        # Draw status text
        status_surf = menu_font.render(status_text, True, status_color)
        status_rect = status_surf.get_rect(center=(SCREEN_WIDTH // 2, 310))
        screen.blit(status_surf, status_rect)

        # Draw IP/Port info
        info_text = f"IP: {multiplayer_target_ip}   Port: {multiplayer_port}   Name: {player_name}"
        info_surf = input_font.render(info_text, True, GRAY)
        info_rect = info_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30))
        screen.blit(info_surf, info_rect)


        for button in buttons: button.draw(screen)

        pygame.display.flip()
        clock.tick(30) # Lower tick rate in lobby is fine


# --- Main Game Loop ---
def main():
    global network_manager, multiplayer_mode # Make network manager accessible globally

    pygame.init()
    # Load settings initially
    settings.load_settings()

    # Ensure 'assets' directory exists
    assets_dir = "assets"
    if not os.path.exists(assets_dir):
        try:
            os.makedirs(assets_dir)
            print(f"Created '{assets_dir}' directory for blueprints.")
        except OSError as e:
            print(f"Error creating directory {assets_dir}: {e}")
            sys.exit(1)

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("PySpaceFlight - Main Menu")
    clock = pygame.time.Clock()

    current_state = STATE_MAIN_MENU
    blueprint_to_launch = None
    running = True

    while running:
        # --- Handle State Transitions ---
        if current_state == STATE_MAIN_MENU:
            pygame.display.set_caption("PySpaceFlight - Main Menu")
            result = main_menu(screen, clock)
            if result == "QUIT": running = False
            elif isinstance(result, tuple) and result[0] == "LAUNCH":
                 blueprint_to_launch = result[1]
                 if blueprint_to_launch and os.path.exists(blueprint_to_launch):
                     current_state = STATE_FLIGHT # Single player flight
                 else:
                     print(f"Error: Quick Launch blueprint '{blueprint_to_launch}' not found.")
                     blueprint_to_launch = None; current_state = STATE_MAIN_MENU
            elif isinstance(result, int): current_state = result
            else: current_state = STATE_MAIN_MENU

        elif current_state == STATE_BUILDER:
             # Check if we came from multiplayer lobby
             if multiplayer_mode:
                  pygame.display.set_caption(f"PySpaceFlight - MP Builder ({multiplayer_mode})")
                  # Pass network manager to builder? Or handle launch differently
                  builder_result = builder.run_builder(screen, clock, is_multiplayer=True)
                  if isinstance(builder_result, tuple) and builder_result[0] == "LAUNCH":
                     blueprint_to_launch = builder_result[1]
                     # Don't launch immediately, wait for others or signal readiness
                     # In multiplayer, 'LAUNCH' from builder means 'READY'
                     if blueprint_to_launch and os.path.exists(blueprint_to_launch):
                         # Send blueprint to server/clients
                         if network_manager:
                             try:
                                 with open(blueprint_to_launch, 'r') as f:
                                     bp_json_str = f.read()
                                 bp_msg = {
                                     "type": network.MSG_TYPE_BLUEPRINT,
                                     "pid": network_manager.player_id,
                                     "name": os.path.basename(blueprint_to_launch), # Send filename? Or actual rocket name?
                                     "bp_name": builder.current_blueprint.name, # Send rocket name from blueprint
                                     "json_str": bp_json_str
                                 }
                                 if multiplayer_mode == "CLIENT":
                                     network_manager.send(bp_msg)
                                     print("Sent blueprint to server.")
                                 elif multiplayer_mode == "HOST":
                                     # Host needs to put its own blueprint on queue for flight sim
                                     network_manager.message_queue.put(bp_msg)
                                     # Host also broadcasts its blueprint to clients
                                     network_manager.broadcast(bp_msg)
                                     print("Broadcasted host blueprint.")

                                 # Signal ready to launch
                                 ready_msg = {"type": network.MSG_TYPE_LAUNCH_READY, "pid": network_manager.player_id}
                                 if multiplayer_mode == "CLIENT":
                                     network_manager.send(ready_msg)
                                 elif multiplayer_mode == "HOST":
                                     network_manager.message_queue.put(ready_msg) # Add own ready msg
                                     network_manager.broadcast(ready_msg) # Broadcast readiness
                                 print("Signalled Launch Ready.")
                                 # Transition to Multiplayer Flight Sim (will wait for others there)
                                 current_state = STATE_MULTIPLAYER_FLIGHT

                             except Exception as e:
                                 print(f"Error sending blueprint/ready signal: {e}")
                                 # Go back to lobby on error? Or stay in builder?
                                 current_state = STATE_MULTIPLAYER_LOBBY

                         else: print("Error: No network manager for MP launch.")
                     else: print(f"Error: Invalid MP blueprint '{blueprint_to_launch}'.")

                  elif builder_result == STATE_MAIN_MENU: # Back button in builder
                      current_state = STATE_MULTIPLAYER_LOBBY # Go back to lobby if in MP
                  # Handle other builder results if needed

             else: # Single player builder
                 pygame.display.set_caption("PySpaceFlight - Rocket Editor")
                 builder_result = builder.run_builder(screen, clock)
                 if isinstance(builder_result, tuple) and builder_result[0] == "LAUNCH":
                     blueprint_to_launch = builder_result[1]
                     if blueprint_to_launch and os.path.exists(blueprint_to_launch):
                         current_state = STATE_FLIGHT # Single player flight
                     else:
                         print(f"Error: Invalid SP blueprint '{blueprint_to_launch}'.")
                         blueprint_to_launch = None; current_state = STATE_MAIN_MENU
                 elif builder_result == STATE_MAIN_MENU: current_state = STATE_MAIN_MENU
                 else: current_state = STATE_MAIN_MENU

        elif current_state == STATE_FLIGHT: # Single Player
             if blueprint_to_launch:
                 pygame.display.set_caption(f"PySpaceFlight - Flying SP: {os.path.basename(blueprint_to_launch)}")
                 # Use the original run_simulation
                 flight_sim.run_simulation(screen, clock, blueprint_to_launch)
                 current_state = STATE_MAIN_MENU
                 blueprint_to_launch = None
             else:
                 print("Error: Reached SP FLIGHT state without blueprint.")
                 current_state = STATE_MAIN_MENU

        # --- Multiplayer States ---
        elif current_state == STATE_MULTIPLAYER_LOBBY:
            pygame.display.set_caption("PySpaceFlight - Multiplayer Lobby")
            result = multiplayer_lobby(screen, clock)
            if result == "QUIT": running = False
            elif isinstance(result, int): current_state = result # e.g., STATE_BUILDER or STATE_MAIN_MENU
            else: current_state = STATE_MAIN_MENU # Default back to menu on unexpected lobby exit

        elif current_state == STATE_MULTIPLAYER_FLIGHT:
            pygame.display.set_caption(f"PySpaceFlight - Flying MP ({multiplayer_mode})")
            if network_manager and blueprint_to_launch:
                 # Call the new multiplayer simulation function
                 flight_sim.run_multiplayer_simulation(screen, clock, blueprint_to_launch, network_manager, multiplayer_mode)
                 # After MP flight ends (e.g., Esc pressed), return to lobby
                 current_state = STATE_MULTIPLAYER_LOBBY
                 # Keep blueprint_to_launch? Or clear? Clear for now.
                 blueprint_to_launch = None
            else:
                 print("Error: Reached MP FLIGHT state without network manager or blueprint.")
                 current_state = STATE_MULTIPLAYER_LOBBY
                 # Clean up network if something went wrong
                 if network_manager:
                     network_manager.stop()
                     network_manager = None
                 multiplayer_mode = None


        elif current_state == STATE_SETTINGS:
             pygame.display.set_caption("PySpaceFlight - Settings")
             result = settings_menu(screen, clock)
             if result == "QUIT": running = False
             elif isinstance(result, int): current_state = result
             else: current_state = STATE_MAIN_MENU # Default back

        else:
            print(f"Error: Unknown game state: {current_state}")
            running = False

    # --- Cleanup ---
    if network_manager:
        print("Stopping network manager on exit...")
        network_manager.stop()
        network_manager = None

    pygame.quit()
    print("PySpaceFlight Exited.")
    sys.exit()


if __name__ == '__main__':
    main()