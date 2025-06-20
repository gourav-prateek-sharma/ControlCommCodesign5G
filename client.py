# client.py
import socket
import json
import math # Corrected import
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from inverted_pendulum import InvertedPendulum
import select
import time

# --- Configuration ---
HOST = '127.0.0.1'
PORT = 65432
DT = 0.01  # 10 ms periodicity

# --- Global objects ---
# Start the pendulum slightly off-center to see the controller work
pendulum = InvertedPendulum(initial_state=[0.0, 0.0, 0.05, 0.0])  # Smaller initial angle
sock = None

# --- Stability Metrics Tracking ---
metrics = {
    'angles': [],
    'positions': [],
    'times': [],
    'forces': [],
    'fall_time': None,
    'settled': False,
    'settling_time': None,
    'peak_overshoot': 0.0,
    'steady_state_error': None,
    'max_angle': 0.0,
    'ISE': 0.0,
    'ITAE': 0.0,
    'control_effort': 0.0
}

SETTLING_BAND = math.radians(2)  # ±2% band in radians (approx 2 deg)
SETTLING_TIME_WINDOW = 1.0  # seconds to consider as 'settled'


def update_metrics(state, t):
    x, x_dot, theta, theta_dot = state
    metrics['angles'].append(theta)
    metrics['positions'].append(x)
    metrics['times'].append(t)
    force = network_control_step.last_action if hasattr(network_control_step, 'last_action') else 0.0
    metrics['forces'].append(force)
    abs_theta = abs(theta)
    if abs_theta > metrics['max_angle']:
        metrics['max_angle'] = abs_theta
    if abs_theta > metrics['peak_overshoot']:
        metrics['peak_overshoot'] = abs_theta
    # Update ISE, ITAE, and control effort (using rectangle rule)
    if len(metrics['angles']) > 1:
        dt = metrics['times'][-1] - metrics['times'][-2]
        metrics['ISE'] += theta**2 * dt
        metrics['ITAE'] += t * abs(theta) * dt
        metrics['control_effort'] += force**2 * dt


def compute_settling_time():
    """Compute settling time: time after which angle stays within SETTLING_BAND for SETTLING_TIME_WINDOW."""
    angles = metrics['angles']
    times = metrics['times']
    for i in range(len(angles)):
        if all(abs(a) < SETTLING_BAND for a in angles[i:i+int(SETTLING_TIME_WINDOW/DT)]):
            return times[i]
    return None


def compute_steady_state_error():
    """Compute steady-state error as mean of last 1 second of angles."""
    N = int(1.0 / DT)
    if len(metrics['angles']) < N:
        return None
    return sum(metrics['angles'][-N:]) / N


def print_metrics():
    print("\n--- Stability Metrics ---")
    print(f"Settling Time: {metrics['settling_time'] if metrics['settling_time'] is not None else 'Not settled'} s")
    print(f"Peak Overshoot: {math.degrees(metrics['peak_overshoot']):.2f} deg")
    print(f"Steady-State Error: {math.degrees(metrics['steady_state_error']) if metrics['steady_state_error'] is not None else 'N/A'} deg")
    print(f"Max Angle from Vertical: {math.degrees(metrics['max_angle']):.2f} deg")
    print(f"Fall Time: {metrics['fall_time'] if metrics['fall_time'] is not None else 'Did not fall'} s")
    print(f"ISE (Integral of Squared Error): {metrics['ISE']:.4f}")
    print(f"ITAE (Integral of Time-weighted Abs Error): {metrics['ITAE']:.4f}")
    print(f"Control Effort (∫u²dt): {metrics['control_effort']:.4f}")


def network_control_step():
    """Performs one step of the networked control loop without artificial delay."""
    global sock, pendulum
    if not hasattr(network_control_step, 'last_action'):
        network_control_step.last_action = 0.0

    # 1. Get current state from the simulation
    current_state = pendulum.get_state()

    # 2. Send state to the server, with newline delimiter
    state_data = (json.dumps(current_state) + '\n').encode('utf-8')
    sock.sendall(state_data)

    # 3. Non-blocking receive for actuation from the server
    ready = select.select([sock], [], [], 0)
    if ready[0]:
        actuation_data = sock.recv(1024)
        if actuation_data:
            # Split by newline and parse the first complete message
            messages = actuation_data.decode('utf-8').split('\n')
            for msg in messages:
                if msg.strip():
                    actuation = json.loads(msg)
                    network_control_step.last_action = actuation['force']
                    break
    # If not ready, use last_action

    # 4. Apply the last received force to the simulation
    pendulum.step(network_control_step.last_action, DT)

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
    t = frame * DT
    success, state = network_control_step()
    update_metrics(state, t)
    
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
        metrics['fall_time'] = t
        ani.event_source.stop()
        metrics['settling_time'] = compute_settling_time()
        metrics['steady_state_error'] = compute_steady_state_error()
        print_metrics()
    # Check for settling (if not already settled and not fallen)
    if not metrics['settled'] and metrics['fall_time'] is None:
        settling_time = compute_settling_time()
        if settling_time is not None:
            metrics['settled'] = True
            metrics['settling_time'] = settling_time
            metrics['steady_state_error'] = compute_steady_state_error()
            print("System settled at t = {:.2f} s".format(settling_time))
            print_metrics()
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