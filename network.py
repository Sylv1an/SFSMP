# network.py
import socket
import threading
import json
import queue
import time

BUFFER_SIZE = 4096  # Adjust as needed

# --- Message Types ---
MSG_TYPE_ERROR = "ERROR"
MSG_TYPE_CONNECT_OK = "CONNECT_OK" # Server confirms connection, sends assigned ID
MSG_TYPE_PLAYER_JOINED = "PLAYER_JOINED" # Server informs clients about a new player
MSG_TYPE_PLAYER_LEFT = "PLAYER_LEFT"   # Server informs clients about a player leaving
MSG_TYPE_REQ_PLAYER_LIST = "REQ_PLAYER_LIST" # Client asks for current players
MSG_TYPE_PLAYER_LIST = "PLAYER_LIST"     # Server sends current player list {id: name}
MSG_TYPE_REQ_BLUEPRINT = "REQ_BLUEPRINT" # Client asks for a specific player's blueprint
MSG_TYPE_BLUEPRINT = "BLUEPRINT"       # Client/Server sends blueprint {pid: id, name: name, json_str: str}
MSG_TYPE_LAUNCH_READY = "LAUNCH_READY"   # Client informs server they are ready to launch
MSG_TYPE_REQ_GAME_STATE = "REQ_GAME_STATE" # Client asks for state of all other rockets
MSG_TYPE_GAME_STATE = "GAME_STATE"     # Server sends state of all rockets {pid: {state_data}}
MSG_TYPE_ACTION = "ACTION"           # Client sends an action {pid: id, action: type, data: {}}
MSG_TYPE_ROCKET_UPDATE = "ROCKET_UPDATE" # Server broadcasts action/state changes {pid: id, action: type, data: {}}
MSG_TYPE_PING = "PING"               # Basic keep-alive or latency check
MSG_TYPE_PONG = "PONG"
MSG_TYPE_SET_NAME = "SET_NAME"         # Client sets its player name {name: str}

# --- Base Network Class ---
class NetworkBase:
    def __init__(self):
        self.message_queue = queue.Queue() # Messages received from network thread
        self._stop_event = threading.Event()
        self._threads = []

    def _start_thread(self, target, args=()):
        thread = threading.Thread(target=target, args=args)
        thread.daemon = True # Allow program to exit even if threads are running
        thread.start()
        self._threads.append(thread)

    def _receive_messages(self, sock):
        """Handles receiving data and putting messages onto the queue."""
        sock.settimeout(0.5) # Check for stop signal periodically
        buffer = b""
        while not self._stop_event.is_set():
            try:
                data = sock.recv(BUFFER_SIZE)
                if not data:
                    # Connection closed by the other side
                    print(f"Connection closed by remote host ({sock.getpeername()}).")
                    self.message_queue.put({"type": MSG_TYPE_ERROR, "data": "Connection closed"})
                    break

                buffer += data
                # Process buffer for complete JSON messages (separated by newline)
                while b'\n' in buffer:
                    message_data, buffer = buffer.split(b'\n', 1)
                    if message_data:
                        try:
                            # Decode assuming UTF-8, ignore errors for robustness
                            message_str = message_data.decode('utf-8', errors='ignore')
                            if message_str.strip(): # Ensure it's not just whitespace
                                message = json.loads(message_str)
                                self.message_queue.put(message)
                        except json.JSONDecodeError:
                            print(f"Warning: Received invalid JSON data: {message_data[:100]}") # Print first 100 bytes
                        except Exception as e:
                            print(f"Error processing received data: {e}")

            except socket.timeout:
                continue # Just loop again to check stop_event
            except socket.error as e:
                print(f"Socket error during receive: {e}")
                self.message_queue.put({"type": MSG_TYPE_ERROR, "data": f"Socket error: {e}"})
                break
            except Exception as e:
                 print(f"Unexpected error during receive: {e}")
                 self.message_queue.put({"type": MSG_TYPE_ERROR, "data": f"Receive error: {e}"})
                 break
        print("Receive thread finished.")

    def send_message(self, sock, message_dict):
        """Sends a dictionary as a JSON string, newline terminated."""
        try:
            message_json = json.dumps(message_dict)
            sock.sendall(message_json.encode('utf-8') + b'\n')
            # print(f"Sent: {message_dict}") # Debug: Log sent messages
            return True
        except socket.error as e:
            print(f"Socket error during send: {e}")
            self.message_queue.put({"type": MSG_TYPE_ERROR, "data": f"Send error: {e}"})
            return False
        except Exception as e:
            print(f"Error encoding message: {e}")
            return False

    def stop(self):
        """Signals all threads to stop and waits for them."""
        print("Network stopping...")
        self._stop_event.set()
        # Give threads a moment to notice the stop event
        time.sleep(0.6)
        # Join remaining threads (optional, depends if errors are critical)
        # for thread in self._threads:
        #     thread.join(timeout=1.0) # Add timeout to prevent hanging
        print("Network stopped.")


# --- Server Class ---
class Server(NetworkBase):
    def __init__(self, host='0.0.0.0', port=65432):
        super().__init__()
        self.host = host
        self.port = port
        self.server_socket = None
        self.clients = {} # {socket: {'id': player_id, 'name': name, 'address': addr}}
        self.client_handlers = {} # {socket: thread}
        self.next_player_id = 1

    def start(self):
        """Starts the server listening for connections."""
        self._stop_event.clear() # Ensure stop event is clear before starting
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen()
            self.server_socket.settimeout(1.0) # Timeout for accept
            print(f"Server listening on {self.host}:{self.port}")
            self._start_thread(self._accept_connections)
            return True
        except socket.error as e:
            print(f"Failed to start server: {e}")
            self.message_queue.put({"type": MSG_TYPE_ERROR, "data": f"Server start failed: {e}"})
            self.server_socket = None
            return False

    def _accept_connections(self):
        """Thread target to accept incoming client connections."""
        while not self._stop_event.is_set():
            try:
                client_socket, addr = self.server_socket.accept()
                print(f"Connection accepted from {addr}")
                # Assign player ID and store client info
                player_id = self.next_player_id
                self.next_player_id += 1
                player_name = f"Pilot_{player_id}" # Default name
                self.clients[client_socket] = {'id': player_id, 'name': player_name, 'address': addr}

                # Send confirmation and assigned ID to the new client
                connect_ok_msg = {"type": MSG_TYPE_CONNECT_OK, "pid": player_id, "name": player_name}
                if not self.send_message(client_socket, connect_ok_msg):
                     print(f"Failed to send connect confirmation to {addr}. Closing connection.")
                     client_socket.close()
                     del self.clients[client_socket]
                     continue # Skip starting handler thread

                # Start a handler thread for this client
                handler_thread = threading.Thread(target=self._handle_client, args=(client_socket,))
                handler_thread.daemon = True
                handler_thread.start()
                self.client_handlers[client_socket] = handler_thread

                # Notify existing clients about the new player
                join_msg = {"type": MSG_TYPE_PLAYER_JOINED, "pid": player_id, "name": player_name}
                self.broadcast(join_msg, exclude_socket=client_socket) # Don't send to the new player

                # Add join message to server's own queue for game logic update
                self.message_queue.put(join_msg)


            except socket.timeout:
                continue # Check stop event again
            except socket.error as e:
                if not self._stop_event.is_set(): # Ignore errors during shutdown
                    print(f"Error accepting connections: {e}")
                break # Exit loop on other errors
        print("Server accept thread finished.")
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None

    def _handle_client(self, client_socket):
        """Thread target to handle messages from a single client."""
        buffer = b""
        client_info = self.clients.get(client_socket)
        player_id = client_info['id'] if client_info else 'Unknown'

        client_socket.settimeout(0.5) # Check stop signal periodically

        while not self._stop_event.is_set():
            try:
                data = client_socket.recv(BUFFER_SIZE)
                if not data:
                    print(f"Client {player_id} ({client_info.get('address', 'N/A')}) disconnected.")
                    break # Exit loop if connection closed

                buffer += data
                while b'\n' in buffer:
                    message_data, buffer = buffer.split(b'\n', 1)
                    if message_data:
                        try:
                            message_str = message_data.decode('utf-8', errors='ignore')
                            if message_str.strip():
                                message = json.loads(message_str)
                                # Add player ID to the message before putting on queue
                                message['pid'] = player_id
                                # print(f"Server RX from {player_id}: {message}") # Debug
                                self._process_server_message(client_socket, message)

                        except json.JSONDecodeError:
                            print(f"Warning: Invalid JSON from {player_id}: {message_data[:100]}")
                        except Exception as e:
                            print(f"Error processing client {player_id} message: {e}")

            except socket.timeout:
                continue
            except socket.error as e:
                print(f"Socket error with client {player_id}: {e}")
                break
            except Exception as e:
                print(f"Unexpected error with client {player_id}: {e}")
                break

        # --- Cleanup after client disconnects or error ---
        self._cleanup_client(client_socket)
        print(f"Handler thread for client {player_id} finished.")

    def _process_server_message(self, client_socket, message):
        """Processes messages received by the server before queueing/broadcasting."""
        msg_type = message.get("type")
        player_id = message.get("pid")

        if msg_type == MSG_TYPE_SET_NAME:
            new_name = message.get("name", f"Pilot_{player_id}")
            if client_socket in self.clients:
                old_name = self.clients[client_socket]['name']
                self.clients[client_socket]['name'] = new_name
                print(f"Player {player_id} changed name from '{old_name}' to '{new_name}'")
                # Put on queue for main game logic (maybe update UI)
                self.message_queue.put(message)
                # Broadcast name change to others
                self.broadcast({"type": msg_type, "pid": player_id, "name": new_name}, exclude_socket=client_socket)
            return # Don't broadcast raw SET_NAME

        elif msg_type == MSG_TYPE_REQ_PLAYER_LIST:
            player_list = {info['id']: info['name'] for sock, info in self.clients.items()}
            response = {"type": MSG_TYPE_PLAYER_LIST, "players": player_list}
            self.send_message(client_socket, response)
            return # Don't broadcast request

        elif msg_type == MSG_TYPE_REQ_BLUEPRINT:
            target_pid = message.get("target_pid")
            # Forward request to the target client (or handle if server stores blueprints)
            # For now, assume server doesn't store blueprints, just forwards
            # Or better: put on server queue, let game logic handle finding/sending BP
            self.message_queue.put(message) # Let game logic handle this request
            return

        elif msg_type == MSG_TYPE_BLUEPRINT:
            # Client is sending their blueprint. Broadcast it.
            # Game logic might also want this message on the queue.
            self.message_queue.put(message)
            self.broadcast(message, exclude_socket=client_socket)
            return

        elif msg_type == MSG_TYPE_LAUNCH_READY:
            # Client is ready. Put on queue for game logic.
            self.message_queue.put(message)
            # Could broadcast this readiness to others if needed
            self.broadcast(message, exclude_socket=client_socket)
            return

        elif msg_type == MSG_TYPE_REQ_GAME_STATE:
            # Client requesting current state. Put on queue for game logic.
            # Game logic will need to gather state and send MSG_GAME_STATE back.
            self.message_queue.put(message)
            return

        elif msg_type == MSG_TYPE_ACTION:
            # Action from a client. Put on queue for game logic AND broadcast.
            self.message_queue.put(message)
            # Broadcast as ROCKET_UPDATE for clarity on client side
            update_msg = message.copy()
            update_msg["type"] = MSG_TYPE_ROCKET_UPDATE
            self.broadcast(update_msg, exclude_socket=client_socket)
            return

        elif msg_type == MSG_TYPE_PING:
             # Respond directly to the specific client
             self.send_message(client_socket, {"type": MSG_TYPE_PONG, "pid": player_id})
             return # Don't broadcast ping/pong

        # --- Default: Put message on queue for game logic and maybe broadcast ---
        # If message wasn't handled specifically above, put it on the main queue
        self.message_queue.put(message)
        # Decide if it needs broadcasting (e.g., generic events) - add logic here if needed

    def _cleanup_client(self, client_socket):
        """Removes client data and notifies others."""
        # --- FIX: Handle the host's conceptual entry (key is None) ---
        if client_socket is None:
            # This is the host's entry, just remove it from the dictionary
            client_info = self.clients.pop(client_socket, None)
            if client_info:
                print(f"Removed host conceptual entry {client_info.get('id')}")
            # Do not try to close socket or broadcast leave message for the host itself
            return
        # --- End Fix ---

        # Original logic for actual client sockets:
        client_info = self.clients.pop(client_socket, None)
        self.client_handlers.pop(client_socket, None)
        try:
            # Check if it's a valid socket before attempting to close
            if hasattr(client_socket, 'close') and callable(client_socket.close):
                client_socket.close()
        except socket.error:
            pass  # Ignore errors closing already closed socket

        if client_info:
            player_id = client_info['id']
            player_name = client_info['name']
            print(f"Cleaned up client {player_id} ({player_name})")
            leave_msg = {"type": MSG_TYPE_PLAYER_LEFT, "pid": player_id, "name": player_name}
            self.message_queue.put(leave_msg)  # Notify server game logic
            self.broadcast(leave_msg)  # Notify remaining clients


    def broadcast(self, message_dict, exclude_socket=None):
        """Sends a message to all connected clients except the excluded one."""
        disconnected_clients = []
        # Iterate over a copy of the client sockets
        for sock in list(self.clients.keys()):
            if sock != exclude_socket:
                if not self.send_message(sock, message_dict):
                    # Mark for cleanup if sending fails
                    print(f"Failed to send broadcast to client {self.clients.get(sock, {}).get('id', 'N/A')}. Marking for removal.")
                    disconnected_clients.append(sock)

        # Cleanup clients that failed during broadcast
        for sock in disconnected_clients:
            self._cleanup_client(sock)


    def stop(self):
        """Stops the server and closes all connections."""
        super().stop() # Signal threads via event
        # Close client sockets first
        print("Closing client connections...")
        for sock in list(self.clients.keys()):
             self._cleanup_client(sock) # Use cleanup ensures proper removal and notification
        # Close server socket
        if self.server_socket:
            print("Closing server socket...")
            try:
                 self.server_socket.close()
                 self.server_socket = None
            except socket.error as e:
                 print(f"Error closing server socket: {e}")
        print("Server stopped fully.")

# --- Client Class ---
class Client(NetworkBase):
    def __init__(self):
        super().__init__()
        self.socket = None
        self.player_id = None
        self.player_name = None
        self.is_connected = False

    def connect(self, host, port, player_name="Pilot"):
        """Connects to the server."""
        self._stop_event.clear()
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0) # Connection timeout
            self.socket.connect((host, port))
            self.socket.settimeout(None) # Reset timeout after connection
            print(f"Connected to server at {host}:{port}")
            self.is_connected = True

            # Start receiver thread
            self._start_thread(self._receive_messages, args=(self.socket,))

            # Send initial name (server might assign ID first)
            # The server will send CONNECT_OK with the assigned ID
            self.send_message(self.socket, {"type": MSG_TYPE_SET_NAME, "name": player_name})

            return True
        except socket.timeout:
            print(f"Connection timed out to {host}:{port}")
            self.message_queue.put({"type": MSG_TYPE_ERROR, "data": "Connection timed out"})
            self.socket = None
            return False
        except socket.error as e:
            print(f"Failed to connect to server {host}:{port}: {e}")
            self.message_queue.put({"type": MSG_TYPE_ERROR, "data": f"Connection failed: {e}"})
            self.socket = None
            return False

    def send(self, message_dict):
        """Sends a message to the server."""
        if not self.is_connected or not self.socket:
            print("Cannot send message: Client not connected.")
            return False
        # Add player ID if known (server usually adds it on receive side)
        # if self.player_id is not None:
        #     message_dict['pid'] = self.player_id
        return self.send_message(self.socket, message_dict)

    def stop(self):
        """Disconnects from the server and stops threads."""
        super().stop()
        self.is_connected = False
        if self.socket:
            print("Closing client socket...")
            try:
                self.socket.shutdown(socket.SHUT_RDWR) # Graceful shutdown
            except socket.error: pass # Ignore if already closed
            try:
                self.socket.close()
            except socket.error: pass # Ignore errors closing already closed socket
            self.socket = None
            self.player_id = None
            self.player_name = None
        print("Client stopped fully.")