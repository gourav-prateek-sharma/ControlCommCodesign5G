# server.py
import socket
import json
from pid_controller import PID
from inverted_pendulum import InvertedPendulum  # Assuming this is the correct import for your InvertedPendulum class
import numpy as np
from control import lqr

# --- Configuration ---
HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)
DT = 0.1          # Simulation time step, must match client

def run_server():
    # --- Controller Setup (Gains from your reference) ---
    # We use two controllers: one for position and one for angle.

    # 1. Position Controller (PD Controller)
    # This controller tries to keep the cart at x=0.
    # It's a Proportional-Derivative (PD) controller.
    x_sp = 0.0  # Position set-point (target)
    kp_x = 0.18  # Increased proportional gain
    kd_x = -0.05  # Reduced derivative gain
    x_bias = 0.01  # Small bias to push the cart toward the center

    # 2. Angle Controller (PID Controller)
    # This controller tries to keep the pole upright (theta=0).
    theta_sp = 0.0 # Angle set-point (target)
    # We can reuse our PID class for this one.
    angle_pid = PID(
        Kp=20.0,  # Reduced proportional gain
        Ki=6.0,   # Reduced integral gain
        Kd=3.0,   # Increased derivative gain
        set_point=theta_sp, 
        dt=DT
    )

    # Maximum force to prevent instability
    F_max = 20.0

    # Initialize the pendulum system
    pendulum = InvertedPendulum(initial_state=[0.0, 0.0, 0.01, 0.0])  # Centered cart and small initial angle

    # Define system matrices for linearized dynamics
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, -pendulum.pole_mass_length * pendulum.gravity / pendulum.total_mass, 0],
        [0, 0, 0, 1],
        [0, 0, (pendulum.total_mass * pendulum.gravity) / (pendulum.length * pendulum.total_mass), 0]
    ])

    B = np.array([[0], [1 / pendulum.total_mass], [0], [-1 / (pendulum.length * pendulum.total_mass)]])

    # Weight matrices for LQR
    Q = np.diag([1, 1, 10, 1])  # Penalize angle more heavily
    R = np.array([[0.1]])       # Penalize control effort

    # Compute LQR gain matrix
    K = lqr_control(A, B, Q, R)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")
        print(f"Control Gains: Kp_x={kp_x}, Kd_x={kd_x} | Kp_th={angle_pid.Kp}, Ki_th={angle_pid.Ki}, Kd_th={angle_pid.Kd}")
        
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                # 1. Receive state from client
                data = conn.recv(1024)
                if not data:
                    print("Client disconnected.")
                    break
                
                # Decode the state from JSON
                state = json.loads(data.decode('utf-8'))
                x, x_dot, theta, theta_dot = state
                
                # --- 2. Compute Actuation (Force) ---
                
                # Position controller output (PD)
                # Force proportional to position error and velocity
                force_x = kp_x * (x_sp - x) + kd_x * x_dot + x_bias
                
                # Angle controller output (PID)
                # We use the PID class to compute the force needed to correct the angle.
                # The input to the controller is the current angle 'theta'.
                force_theta = angle_pid.compute(theta)

                # Total Force Calculation
                # The final force is the sum of the two controller outputs.
                # The negative sign is crucial and standard in this control formulation.
                total_force = -(force_x + force_theta)

                # Clamp the total force to the maximum allowed value
                if abs(total_force) > F_max:
                    total_force = F_max if total_force > 0 else -F_max

                print(f"State[x={x:.2f}, θ={theta:.3f}] -> Force[pos={-force_x:.2f}, ang={-force_theta:.2f}] -> Total={total_force:.2f}")

                # LQR control logic
                state_vector = np.array([x, x_dot, theta, theta_dot])
                lqr_force = -np.dot(K, state_vector)

                # Introduce a small perturbation to the LQR force
                perturbation = np.random.uniform(-0.5, 0.5)  # Random noise in range [-0.5, 0.5]
                lqr_force += perturbation

                # Clamp the perturbed LQR force to the maximum allowed value
                if abs(lqr_force) > F_max:
                    lqr_force = F_max if lqr_force > 0 else -F_max

                print(f"State[x={x:.2f}, θ={theta:.3f}] -> Perturbed LQR Force={lqr_force:.2f}")

                # 3. Send actuation back to client
                actuation_data = {'force': lqr_force}
                conn.sendall(json.dumps(actuation_data).encode('utf-8'))

def lqr_control(A, B, Q, R):
    """Compute the LQR gain matrix."""
    K, _, _ = lqr(A, B, Q, R)
    return np.array(K).flatten()

if __name__ == '__main__':
    run_server()