import matplotlib.pyplot as plt
import numpy as np
from device import RLDevice
from simulation import run_simulation

num_devices = 10
num_steps = 1000
num_runs = 10

# --- BASELINE ---
t_base, c_base, _ = run_simulation(
    num_devices=num_devices,
    time_units=num_steps,
    use_rl=False, 
)

# --- RL (learning across runs) ---
rl_devices = [RLDevice(i) for i in range(num_devices)]

rl_runs = []  # store all RL runs

for run in range(num_runs):
    t_rl, c_rl, rl_devices = run_simulation(
        num_devices=num_devices,
        time_units=num_steps,
        use_rl=True,
        devices=rl_devices
    )
    rl_runs.append((t_rl, c_rl))

# ==============================
# 📊 PLOT 1: RL learning across runs
# ==============================
plt.figure()
for i, (t_rl, _) in enumerate(rl_runs):
    plt.plot(t_rl, linestyle="--", label=f"RL Run {i+1}")
plt.plot(t_base, linewidth=3, label="Baseline")
plt.legend()
plt.title("RL Learning Across Runs vs Baseline")
plt.xlabel("Time")
plt.ylabel("Successful Transmissions")
plt.grid(True)
plt.show()

# ==============================
# 📊 PLOT 2: Final throughput comparison
# ==============================
final_t_rl, final_c_rl = rl_runs[-1]

plt.figure()
plt.plot(t_base, label="Baseline")
plt.plot(final_t_rl, label="RL (Final Run)")
plt.legend()
plt.title("Throughput Comparison")
plt.xlabel("Time")
plt.ylabel("Successful Transmissions")
plt.grid(True)
plt.show()

# ==============================
# 📊 PLOT 3: Collision comparison
# ==============================
plt.figure()
plt.plot(c_base, label="Baseline")
plt.plot(final_c_rl, label="RL (Final Run)")
plt.legend()
plt.title("Collision Comparison")
plt.xlabel("Time")
plt.ylabel("Collisions")
plt.grid(True)
plt.show()