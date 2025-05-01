import socket
import pickle
import gymnasium as gym
import time
import numpy as np

UDP_IP = "127.0.0.1"
SEND_PORT = 5005   # Send state to controller
RECV_PORT = 5006   # Receive action from controller
dt = 0.05          # 20 Hz control loop

# Create UDP sockets
send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_sock.bind((UDP_IP, RECV_PORT))
recv_sock.settimeout(1.0)

# Initialize environment
env = gym.make("CartPole-v0", render_mode="human")
obs, _ = env.reset()

while True:
    # Send current observation
    send_time = time.time()
    data = pickle.dumps((obs, send_time))
    send_sock.sendto(data, (UDP_IP, SEND_PORT))

    try:
        # Wait for control action
        action_data, _ = recv_sock.recvfrom(1000)
        recv_time = time.time()
        action = pickle.loads(action_data)

        # Apply control
        obs, reward, terminated, truncated, _ = env.step(action)

        # RTT logging
        rtt = (recv_time - send_time) * 1000  # in ms
        print(f"RTT: {rtt:.2f} ms, Action: {action}, State: {obs}")

        if terminated or truncated:
            obs, _ = env.reset()

    except socket.timeout:
        print("No control input received. Skipping step.")
    
    time.sleep(dt)
