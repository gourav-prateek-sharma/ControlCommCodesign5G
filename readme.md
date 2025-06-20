# Inverted Pendulum Networked Control System

This project implements a networked inverted pendulum (cart-pole) control system in Python, supporting both GUI visualization and headless metric logging. The system uses a client-server architecture for real-time control and simulation.

## Requirements
- Python 3.x
- `numpy`, `matplotlib`, `control`

Install dependencies:
```bash
pip install numpy matplotlib control
```

## Usage

### 1. Start the Server
Open a terminal and run:
```bash
python server.py
```

The server will listen for client connections and send control actions.

### 2. Run the Client
Open another terminal and run one of the following:

#### GUI Mode (with animation)
```bash
python client.py
```

#### Headless Mode (no GUI, logs metrics to CSV)
```bash
python client.py --no-gui --sim-time 20 --metrics-file control_metrics.csv
```
- `--no-gui`: Run without animation, log metrics to file
- `--sim-time`: Duration of simulation in seconds (default: 10)
- `--metrics-file`: Output CSV file for metrics (default: metrics_log.csv)

## Output
- In GUI mode, you will see a real-time animation and dynamic metrics.
- In headless mode, metrics are saved to the specified CSV file and summary statistics are printed at the end.

## Metrics Tracked
- Settling Time
- Peak Overshoot
- Steady-State Error
- Maximum Angle from Vertical
- Fall Time
- ISE (Integral of Squared Error)
- ITAE (Integral of Time-weighted Absolute Error)
- Control Effort (∫u²dt)

## Notes
- Ensure the server is running before starting the client.
- The simulation time step and control delay are set in the code and should match between client and server.

