# pid_controller.py

class PID:
    def __init__(self, Kp, Ki, Kd, set_point=0.0, dt=0.01):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.set_point = set_point
        self.dt = dt
        
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, current_value):
        """Calculates PID value for a given current value."""
        error = self.set_point - current_value
        
        # Proportional term
        P_term = self.Kp * error
        
        # Integral term (with anti-windup)
        self.integral += error * self.dt
        # Simple anti-windup: clamp the integral term
        self.integral = max(min(self.integral, 20.0), -20.0)
        I_term = self.Ki * self.integral
        
        # Derivative term
        derivative = (error - self.prev_error) / self.dt
        D_term = self.Kd * derivative
        
        # Update previous error
        self.prev_error = error
        
        # Total output
        output = P_term + I_term + D_term
        
        # Clamp the output force to a reasonable range
        return max(min(output, 50.0), -50.0)