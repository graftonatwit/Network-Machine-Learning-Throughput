import matplotlib.pyplot as plt
import numpy as np
import os
from device import Device, RLDevice
from simulation import run_simulation

num_devices = 10
num_steps = 1000
num_runs = 10

# --- BASELINE ---
base_devices = [Device(i) for i in range(num_devices)]  # create baseline devices
base_runs = []  # store all baseline runs
base_averages = []  # store all baseline averages
base_success_rates = []  # store all baseline success rates


for num in range(num_runs):
    t_base, c_base, _ , average_collisions_base, success_rate = run_simulation( 
    num_devices=num_devices,
    time_units=num_steps,
    use_rl=False, 
    devices=base_devices
    )
    base_runs.append((t_base, c_base))
    base_averages.append(average_collisions_base)
    base_success_rates.append(success_rate)

# --- RL (learning across runs) ---
rl_devices = [RLDevice(i) for i in range(num_devices)]  # create RL devices
rl_runs = []  # store all RL runs
rl_averages = []  # store all RL averages
rl_success_rates = []  # store all RL success rates


# --- LOAD P VALUES ---
if os.path.exists("p_values.npy"):
    saved_p_values = np.load("p_values.npy")  # load as NumPy array
    for d, p in zip(rl_devices, saved_p_values):  # set each device's p
        d.p = float(p)

# --- LOAD Q_TABLES ---
if os.path.exists("q_tables.npy"):  # if q_tables.npy exists
    saved_q_tables = np.load("q_tables.npy", allow_pickle=True)  # load q_tables
    for d, qt in zip(rl_devices, saved_q_tables):  # zip rl_devices and q_tables
        d.q_table = qt  # set the q_table for each device


for run in range(num_runs):
    t_rl, c_rl, rl_devices, average_collisions_rl, success_rate = run_simulation(
        num_devices=num_devices,
        time_units=num_steps,
        use_rl=True,
        devices=rl_devices
    )
    rl_runs.append((t_rl, c_rl)) # store each run
    rl_averages.append(average_collisions_rl)
    rl_success_rates.append(success_rate)
    
    
# save p values for all RL devices
p_values = [d.p for d in rl_devices]  # collect current p for each device
np.save("p_values.npy", p_values)  # save to a file


# --- SAVE Q_TABLES ---
q_tables = [d.q_table for d in rl_devices] # store all q_tables
np.save("q_tables.npy", q_tables) # save q_tables

# ==============================
# 📊 PLOT 1: RL learning across runs
# ==============================
plt.figure()
for i, (t_rl, _) in enumerate(rl_runs[-5:], 1): # plot each run start at 5
    plt.plot(t_rl, linestyle="--", label=f"RL Run {len(rl_runs)-5 + i}") # plot each run
for i, (t_base, _) in enumerate(base_runs[-5:], 1): # plot each run start at 5
    plt.plot(t_base, linestyle="-", label=f"Baseline Run {len(base_runs) - 5 + i }")
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
final_t_base, final_c_base = base_runs[-1]
plt.figure()
plt.plot(final_t_base, label="Baseline (Final Run)")
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
plt.plot(final_c_base, label="Baseline (Final Run)")
plt.plot(final_c_rl, label="RL (Final Run)")
plt.legend()
plt.title("Collision Comparison")
plt.xlabel("Time")
plt.ylabel("Collisions")
plt.grid(True)
plt.show()

# ==============================
# 📊 PLOT 4: Average Collisions
# ==============================
plt.figure()
for i, run in enumerate(rl_averages[-5:], 1):
    plt.plot(run, linestyle="--", label=f"RL Run {len(rl_averages)-5+i}")
    
for i, run in enumerate(base_averages[-5:], 1):
    plt.plot(run, linestyle="-", label=f"Baseline Run {len(base_averages)-5+i}")
plt.legend()
plt.title("Collision Comparison")
plt.xlabel("Time")
plt.ylabel("Cumulative Average Collisions Over Time Units (Each Run)")
plt.grid(True)
plt.show()



plt.figure()
for i, run in enumerate(rl_success_rates[-5:], 1):
    plt.plot(run, linestyle="--", label=f"RL Run {len(rl_success_rates)-5+i}")
    
for i, run in enumerate(base_success_rates[-5:], 1):
    plt.plot(run, linestyle="-", label=f"Baseline Run {len(base_success_rates)-5+i}")
plt.legend()
plt.title("Success Rate Comparison")
plt.xlabel("Time Units")
plt.ylabel("Cumulative Success Rates Over Time Units (Each Run)")
plt.grid(True)
plt.show()
