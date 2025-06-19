import numpy as np
from inverted_pendulum import InvertedPendulum
from pid_controller import PID
import matplotlib.pyplot as plt

def simulate_control(Kp, Ki, Kd, simulation_time=5.0, dt=0.01):
    """
    Simulate the pendulum with given PID parameters and return a performance metric.
    Returns None if the pendulum falls, otherwise returns a performance score.
    """
    pendulum = InvertedPendulum([0.0, 0.0, 0.05, 0.0])
    pid = PID(Kp=Kp, Ki=Ki, Kd=Kd, set_point=0.0, dt=dt)
    
    angles = []
    positions = []
    
    # Run simulation
    for t in np.arange(0, simulation_time, dt):
        state = pendulum.get_state()
        x, x_dot, theta, theta_dot = state
        
        # Check if pendulum has fallen
        if abs(theta) > 0.5:  # about 28 degrees
            return None
            
        # Calculate control effort based on angle and angular velocity
        angle_error = theta
        velocity_feedback = theta_dot * 0.5  # Adding damping term
        
        # Combine PID control with direct velocity feedback
        control = pid.compute(angle_error) - velocity_feedback
        
        # Apply force limits more strictly
        force = np.clip(control, -20.0, 20.0)  # More conservative force limits
        
        pendulum.step(force, dt)
        
        angles.append(theta)
        positions.append(x)
    
    # Calculate performance metrics
    angle_stability = np.std(angles)  # Lower is better
    position_stability = np.std(positions)  # Lower is better
    
    # Combined score (lower is better)
    score = angle_stability * 10 + position_stability
    
    return score

def grid_search():
    """
    Perform grid search to find good PID parameters.
    """
    # Define parameter ranges - much more conservative now
    Kp_range = np.linspace(1, 15, 15)     # Very small proportional gains
    Ki_range = np.linspace(0, 0.1, 3)     # Minimal integral action
    Kd_range = np.linspace(1, 5, 5)       # Conservative derivative gains
    
    best_score = float('inf')
    best_params = None
    
    results = []
    
    print("Starting grid search...")
    total_combinations = len(Kp_range) * len(Ki_range) * len(Kd_range)
    current = 0
    
    for Kp in Kp_range:
        for Ki in Ki_range:
            for Kd in Kd_range:
                current += 1
                print(f"\nTesting combination {current}/{total_combinations}")
                print(f"Kp={Kp:.2f}, Ki={Ki:.3f}, Kd={Kd:.2f}")
                
                score = simulate_control(Kp, Ki, Kd)
                
                if score is not None:
                    results.append((Kp, Ki, Kd, score))
                    if score < best_score:
                        best_score = score
                        best_params = (Kp, Ki, Kd)
                        print(f"New best parameters found! Score: {best_score:.4f}")
    
    if best_params:
        print("\nBest parameters found:")
        print(f"Kp = {best_params[0]:.2f}")
        print(f"Ki = {best_params[1]:.3f}")
        print(f"Kd = {best_params[2]:.2f}")
        print(f"Score = {best_score:.4f}")
        
        # Plot results for successful combinations
        if results:
            results = np.array(results)
            plt.figure(figsize=(10, 6))
            plt.scatter(results[:, 0], results[:, 3], alpha=0.5, label='Kp vs Score')
            plt.xlabel('Kp')
            plt.ylabel('Score (lower is better)')
            plt.title('PID Parameter Search Results')
            plt.legend()
            plt.savefig('pid_tuning_results.png')
        
        return best_params
    else:
        print("No stable parameters found!")
        return None

if __name__ == '__main__':
    best_params = grid_search()