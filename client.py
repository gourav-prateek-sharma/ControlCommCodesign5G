# client.py
import socket
import json
import math # Corrected import
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from inverted_pendulum import InvertedPendulum

# --- Configuration ---
HOST = '127.0.0.1'
PORT = 65432
DT = 0.05  # 10 ms periodicity

# --- Global objects ---
# Start the pendulum slightly off-center to see the controller work
pendulum = InvertedPendulum(initial_state=[0.0, 0.0, 0.05, 0.0])  # Smaller initial angle
sock = None

def network_control_step():
    """Performs one step of the networked control loop."""
    global sock, pendulum
    
    # 1. Get current state from the simulation
    current_state = pendulum.get_state()
    
    # 2. Send state to the server
    state_data = json.dumps(current_state).encode('utf-8')
    sock.sendall(state_data)
    
    # 3. Receive actuation from the server
    actuation_data = sock.recv(1024)
    if not actuation_data:
        print("Server disconnected.")
        return False, None
        
    actuation = json.loads(actuation_data.decode('utf-8'))
    force = actuation['force']
    
    # 4. Apply actuation to the simulation
    pendulum.step(force, DT)
    
    return True, pendulum.get_state()

# --- Visualization Setup ---
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-0.5, 1.5)
ax.set_title("Networked Control System: Inverted Pendulum (Cascade Control)")
track = ax.axhline(0, color='gray', lw=2)
cart, = ax.plot([], [], 's', ms=20, color='royalblue', label='Cart')
pole, = ax.plot([], [], 'o-', lw=3, color='orangered', markersize=6, label='Pole')
# Add a text display for the state
state_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, verticalalignment='top')


def init_animation():
    cart.set_data([], [])
    pole.set_data([], [])
    state_text.set_text('')
    return cart, pole, state_text

def update_animation(frame):
    """This function is called by FuncAnimation to update the plot."""
    success, state = network_control_step()
    
    if not success:
        ani.event_source.stop()
        return cart, pole, state_text

    x, x_dot, theta, theta_dot = state
    
    # Drawing logic
    # The pendulum model uses 'length' as half the pole length
    pole_length = pendulum.length * 2
    pole_x = [x, x + pole_length * math.sin(theta)]
    pole_y = [0, pole_length * math.cos(theta)]
    
    cart.set_data([x], [0])
    pole.set_data(pole_x, pole_y)
    
    # Update state text
    state_str = f'x = {x:.2f} m\nθ = {math.degrees(theta):.2f}°'
    state_text.set_text(state_str)

    # Failure condition (pole falls too far)
    if abs(theta) > math.radians(45):
        print("Pendulum has fallen!")
        ani.event_source.stop()

    return cart, pole, state_text

# --- Main Execution ---
if __name__ == '__main__':
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"Connecting to server at {HOST}:{PORT}...")
        sock.connect((HOST, PORT))
        print("Connection successful!")
        
        ani = animation.FuncAnimation(fig, update_animation,
                                      frames=None,
                                      init_func=init_animation,
                                      blit=True,
                                      interval=DT * 1000,
                                      repeat=False)
        plt.legend()
        plt.show()

    except ConnectionRefusedError:
        print("\n[ERROR] Connection refused. Is the server script running in another terminal?")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if sock:
            print("Closing connection.")
            sock.close()