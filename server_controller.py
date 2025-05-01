import socket
import pickle
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

UDP_IP = "127.0.0.1"
SEND_PORT = 5006  # Send action to client
RECV_PORT = 5005  # Receive state from client

# Load environment for shape reference only (no simulation)
dummy_env = gym.make("CartPole-v0")
model = PPO("MlpPolicy", dummy_env, verbose=0)

# Create UDP sockets
recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_sock.bind((UDP_IP, RECV_PORT))

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("Controller server running...")

while True:
    # Receive state
    data, addr = recv_sock.recvfrom(1000)
    obs, send_time = pickle.loads(data)
    
    t1 = time.time()
    # Predict action
    action, _ = model.predict(obs, deterministic=True)
    t2 = time.time()
    print((t2-t1)*1000)
    # Send action back
    action_data = pickle.dumps(action)
    send_sock.sendto(action_data, (UDP_IP, SEND_PORT))
