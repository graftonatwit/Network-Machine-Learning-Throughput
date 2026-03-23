import matplotlib.pyplot as plt
import numpy as np
import os
from device import Device, RLDevice, SarmaDevice
from simulation import run_simulation

num_devices = 10
num_steps = 1000
num_runs = 2

# --- BASELINE ---

base_runs = []  # store all baseline runs
base_averages = []  # store all baseline averages
base_success_rates = []  # store all baseline success rates

for num in range(num_runs):
    base_devices = [Device(i) for i in range(num_devices)]  # create baseline devices
    t_base, c_base, _ , average_collisions_base, success_rate, _ = run_simulation( 
    num_devices=num_devices,
    time_units=num_steps,
    use_rl=False, 
    use_sarma=False,
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
rl_epsilon = []  # store all RL epsilons


# --- LOAD P VALUES ---
if os.path.exists("p_values.npy"):
    saved_p_values = np.load("p_values.npy")  # load as NumPy array
    for d, p in zip(rl_devices, saved_p_values):  # set each device's p
        d.p = float(p)  # convert to float

# --- LOAD Q_TABLES ---
if os.path.exists("q_tables.npy"):  # if q_tables.npy exists
    saved_q_tables = np.load("q_tables.npy", allow_pickle=True)  # load q_tables
    for d, qt in zip(rl_devices, saved_q_tables):  # zip rl_devices and q_tables
        d.q_table = qt  # set the q_table for each device
        


# --- LOAD EPSILON ---
if os.path.exists("epsilon_final.npy"):
    saved_epsilon = np.load("epsilon_final.npy")
    for d, e in zip(rl_devices, saved_epsilon):
        d.epsilon = float(e)


# --- RUN RL SIMULATION ---
for run in range(num_runs):
    t_rl, c_rl, rl_devices, average_collisions_rl, success_rate, epsilon = run_simulation(
        num_devices=num_devices,
        time_units=num_steps,
        use_rl=True,
        use_sarma=False,
        devices=rl_devices
    )
    rl_runs.append((t_rl, c_rl)) # store each run
    rl_averages.append(average_collisions_rl)
    rl_success_rates.append(success_rate)
    rl_epsilon.append(epsilon)
    

# save p values for all RL devices
p_values = [d.p for d in rl_devices]  # collect current p for each device
np.save("p_values.npy", p_values)  # save to a file


# --- SAVE Q_TABLES ---
q_tables = [d.q_table for d in rl_devices] # store all q_tables
np.save(f"q_tables_run{num_runs}.npy", q_tables) # save q_tables


final_epsilons = [d.epsilon for d in rl_devices] # store all final epsilons
np.save("epsilon_final.npy", final_epsilons) # save


# --- Sarma (learning across runs) ---
sarma_devices = [SarmaDevice(i) for i in range(num_devices)]  # create Sarma devices
sarma_runs = []  # store all Sarma runs
sarma_averages = []  # store all Sarma averages
sarma_success_rates = []  # store all Sarma success rates


# --- LOAD SARMA STATE ---
if os.path.exists("sarma_p_values.npy"):
    saved_p = np.load("sarma_p_values.npy")
    for d, p in zip(sarma_devices, saved_p):
        d.p = float(p)

if os.path.exists("sarma_steps.npy"):
    saved_steps = np.load("sarma_steps.npy")
    for d, s in zip(sarma_devices, saved_steps):
        d.step = float(s)
        

        


for run in range(num_runs):
    t_sarma, c_sarma, sarma_devices, average_collisions_sarma, success_rate, _ = run_simulation(
        num_devices=num_devices,
        time_units=num_steps,
        use_rl=False,
        use_sarma=True,
        devices=sarma_devices
    )
    sarma_runs.append((t_sarma, c_sarma)) # store each run
    sarma_averages.append(average_collisions_sarma) # store each run
    sarma_success_rates.append(success_rate) # store each run
    
    
# --- SAVE SARMA STATE ---
sarma_p_values = [d.p for d in sarma_devices]
sarma_steps = [d.step for d in sarma_devices]

np.save("sarma_p_values.npy", sarma_p_values)
np.save("sarma_steps.npy", sarma_steps)

# ==============================
# 📊 PLOT 1: RL learning across runs
# ==============================
plt.figure()
for i, (t_rl, _) in enumerate(rl_runs, 1): # plot each run 
    plt.plot(t_rl, linestyle="--", label=f"RL Run {len(rl_runs)}") # plot each run
for i, (t_base, _) in enumerate(base_runs, 1): # plot each run
    plt.plot(t_base, linestyle="-", label=f"Baseline Run {len(base_runs)}")
for i, (t_sarma, _) in enumerate(sarma_runs, 1): # plot each run
    plt.plot(t_sarma, linestyle=":", label=f"Sarma Run {len(sarma_runs)}")
plt.legend()
plt.title("RL and Sarma Learning Across Runs vs Baseline")
plt.xlabel("Time")
plt.ylabel("Successful Transmissions")
plt.grid(True)
plt.show()

# ==============================
# 📊 PLOT 2: Final throughput comparison
# ==============================
final_t_rl, final_c_rl = rl_runs[-1]
final_t_base, final_c_base = base_runs[-1]
final_t_sarma, final_c_sarma = sarma_runs[-1]
plt.figure()
plt.plot(final_t_base, label="Baseline (Final Run)")
plt.plot(final_t_rl, label="RL (Final Run)")
plt.plot(final_t_sarma, label="Sarma (Final Run)")
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
plt.plot(final_c_sarma, label="Sarma (Final Run)")
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
for i, run in enumerate(rl_averages, 1):
    plt.plot(run, linestyle="--", label=f"RL Run {i}") # plot each run
    
for i, run in enumerate(base_averages, 1):
    plt.plot(run, linestyle="-", label=f"Baseline Run {i}") # plot each run

for i, run in enumerate(sarma_averages, 1):
    plt.plot(run, linestyle=":", label=f"Sarma Run {i}") # plot each run
plt.legend()
plt.title("Cumulative Average Collisions Over Time")
plt.xlabel("Time")
plt.ylabel("Cumulative Average Collisions Over Time Units (Each Run)")
plt.grid(True)
plt.show()


# ==============================
# 📊 PLOT 5: Success Rates
# ==============================
plt.figure()
for i, run in enumerate(rl_success_rates, 1):
    plt.plot(run, linestyle="--", label=f"RL Run {i}") # plot each run
    
for i, run in enumerate(base_success_rates, 1):
    plt.plot(run, linestyle="-", label=f"Baseline Run {i}") # plot each run

for i, run in enumerate(sarma_success_rates, 1):
    plt.plot(run, linestyle=":", label=f"Sarma Run {i}") # plot each run
plt.legend()
plt.title("Success Rate Comparison")
plt.xlabel("Time Units")
plt.ylabel("Cumulative Success Rates Over Time Units (Each Run)")
plt.grid(True)
plt.show()


# ==============================
# 📊 PLOT 6: Epsilon
# ==============================
plt.figure()
for i, run in enumerate(rl_epsilon, 1):
    plt.plot(run, linestyle="--", label=f"RL Run {i}") # plot each run
plt.legend()
plt.title("Epsilon Over Time")
plt.xlabel("Time")
plt.ylabel("Epsilon")
plt.grid(True)
plt.show()