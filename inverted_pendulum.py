# inverted_pendulum.py
import math

class InvertedPendulum:
    """
    Represents the physics of an inverted pendulum on a cart.
    State vector: [x, x_dot, theta, theta_dot]
      - x: Cart position (m)
      - x_dot: Cart velocity (m/s)
      - theta: Pole angle from vertical (radians)
      - theta_dot: Pole angular velocity (rad/s)
    """
    def __init__(self, initial_state=[0.0, 0.0, 0.05, 0.0]):
        # Physical constants
        self.gravity = 9.8
        self.mass_cart = 0.5
        self.mass_pole = 0.2
        self.total_mass = self.mass_cart + self.mass_pole
        self.length = 0.3  # actually half the pole's length
        self.pole_mass_length = self.mass_pole * self.length
        
        # State vector
        self.state = initial_state

    def get_state(self):
        return self.state

    def step(self, force, dt):
        """
        Updates the state of the pendulum for a given time step 'dt'.
        Uses non-linear dynamics and Euler's method for integration.
        """
        x, x_dot, theta, theta_dot = self.state

        # Equations of motion
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.pole_mass_length * theta_dot**2 * sintheta) / self.total_mass
        
        theta_acc = (self.gravity * sintheta - costheta * temp) / \
                    (self.length * (4.0/3.0 - self.mass_pole * costheta**2 / self.total_mass))
        
        x_acc = temp - self.pole_mass_length * theta_acc * costheta / self.total_mass

        # Euler integration
        x_dot += x_acc * dt
        x += x_dot * dt
        theta_dot += theta_acc * dt
        theta += theta_dot * dt

        self.state = [x, x_dot, theta, theta_dot]
        return self.state