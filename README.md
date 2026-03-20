# Network-Machine-Learning-Throughput
Wireless Network Congestion Control Simulation
Project Overview

This project explores congestion in wireless networks and demonstrates how reinforcement learning (RL) can be used to dynamically adapt device transmission probabilities to improve throughput and reduce collisions. The simulation models multiple devices transmitting packets to a shared access point and compares:

A baseline algorithm using static probability adjustment

A reinforcement learning approach (Q-Learning) where devices learn optimal transmission strategies over time

This project is implemented in Python, with visualization using Matplotlib and numerical computation using NumPy.

Features

Simulates multiple wireless devices competing for the same channel

Baseline algorithm: simple probability adjustment based on success/collision/idle

RL algorithm: each device maintains a Q-table and updates it using Q-Learning

Visualization of:

Successful transmissions (throughput)

Collisions

RL learning across multiple runs

Supports learning persistence across multiple simulation runs

File Structure
.
├── device.py       # Defines Device and RLDevice classes
├── simulation.py   # Contains run_simulation() function
├── plot.py         # Runs simulations and generates graphs
└── README.md       # Project documentation
Installation

Clone the repository:

git clone <your_repo_url>
cd <repo_folder>

Install dependencies:

pip install numpy matplotlib

Python version: 3.8+ recommended

How to Run

Run the simulation and generate plots:

python plot.py

What happens:

The baseline devices simulate network traffic without learning

RL devices simulate network traffic with Q-Learning

Graphs show:

Throughput comparison

Collision comparison

RL learning progression over multiple runs

How the Simulation Works
Baseline Devices

Each device has a transmission probability p

p is updated as:

Success → increase p

Collision → decrease p

Idle → no change

Probabilities are clamped between 0 and 1

Simple, memoryless algorithm

RL Devices

Each device maintains a Q-table for states (idle, success, collision) × actions (decrease, same, increase)

Uses Q-Learning:

Updates Q-values after each step:

Q(s,a) = Q(s,a) + α * (reward + γ * max(Q(next_state)) - Q(s,a))

Chooses actions using epsilon-greedy:

With probability ε → explore (random action)

With probability 1-ε → exploit (best Q-value)

Transmission probability p is adjusted according to chosen action

Small random noise is added to p to avoid synchronized transmissions

Learning persists across multiple runs

Parameters You Can Adjust

num_devices → number of devices in the network

num_steps → number of simulation steps per run

num_runs → number of RL runs to observe learning over time

alpha → learning rate

gamma → discount factor

epsilon → exploration rate

Example Output

Throughput Comparison: baseline vs RL final run

Collision Comparison: baseline vs RL final run

RL Learning Across Runs: shows improvement of RL over multiple runs

Note: RL devices start learning from initial random probabilities and gradually adapt to maximize throughput while avoiding collisions.

Notes / Tips

RL devices initially may perform worse than baseline until they learn

Adjusting alpha, epsilon, and initial p can stabilize learning

Reward shaping is important: collisions should be penalized proportionally to the number of transmitting devices

Small randomness in p prevents devices from synchronizing and colliding
