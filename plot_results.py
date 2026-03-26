import matplotlib.pyplot as plt
import numpy as np
import os
from device import Device, RLDevice, SarmaDevice, HybridDevice, BanditDevice
from simulation import run_simulation

num_devices = 10
num_steps = 1000
num_runs = 2

# --- BASELINE ---

base_runs = []  # store all baseline runs
base_averages = []  # store all baseline cumulative averages of collisions
base_success_rates = []  # store all baseline success rates
base_average_numbers = []  # store all baseline average numbers of collisions


for num in range(num_runs):
    base_devices = [Device(i) for i in range(num_devices)]  # create baseline devices
    t_base, c_base, _ , average_collisions_base, success_rate, _, average_base  = run_simulation( 
    num_devices=num_devices,
    time_units=num_steps,
    use_rl=False, 
    use_sarma=False,
    use_hybrid=False,
    use_bandit=False,
    devices=base_devices
    )
    base_runs.append((t_base, c_base))
    base_averages.append(average_collisions_base)
    base_success_rates.append(success_rate)
    base_average_numbers.append(average_base)


# --- RL (learning across runs) ---
rl_devices = [RLDevice(i) for i in range(num_devices)]  # create RL devices
rl_runs = []  # store all RL runs
rl_averages = []  # store all RL cumulative average collisions
rl_success_rates = []  # store all RL success rates
rl_epsilon = []  # store all RL epsilons
rl_average_numbers = []  # store all RL average numbers of collisions



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
        


# --- RESET EPSILON ---
RESET_EPSILON = False
if RESET_EPSILON:
    for d in rl_devices:
        d.epsilon = 0.1
    np.save("epsilon_final.npy", np.full(len(rl_devices), 0.1))
# ---


        




# --- RUN RL SIMULATION ---
for run in range(num_runs):
    t_rl, c_rl, rl_devices, average_collisions_rl, success_rate, epsilon, average_rl = run_simulation(
        num_devices=num_devices,
        time_units=num_steps,
        use_rl=True,
        use_sarma=False,
        use_hybrid=False,
        use_bandit=False,
        devices=rl_devices
    )
    rl_runs.append((t_rl, c_rl)) # store each run
    rl_averages.append(average_collisions_rl)
    rl_success_rates.append(success_rate)
    rl_epsilon.append(epsilon)
    rl_average_numbers.append(average_rl)

    

# save p values for all RL devices
p_values = [d.p for d in rl_devices]  # collect current p for each device
np.save("p_values.npy", p_values)  # save to a file


# --- SAVE Q_TABLES ---
q_tables = [d.q_table for d in rl_devices] # store all q_tables
np.save(f"q_tables_run{num_runs}.npy", q_tables) # save q_tables


# --- SAVE EPSILON ---
final_epsilons = [d.epsilon for d in rl_devices]
np.save("epsilon_final.npy", final_epsilons) # save


# --- Sarma (learning across runs) ---
sarma_devices = [SarmaDevice(i) for i in range(num_devices)]  # create Sarma devices
sarma_runs = []  # store all Sarma runs
sarma_averages = []  # store all Sarma cumulative averages of collisions
sarma_success_rates = []  # store all Sarma success rates
sarma_average_numbers = []  # store all Sarma average numbers of collisions



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
    t_sarma, c_sarma, sarma_devices, average_collisions_sarma, success_rate, _, average_sarma = run_simulation(
        num_devices=num_devices,
        time_units=num_steps,
        use_rl=False,
        use_sarma=True,
        use_hybrid=False,
        use_bandit=False,
        devices=sarma_devices
    )
    sarma_runs.append((t_sarma, c_sarma)) # store each run
    sarma_averages.append(average_collisions_sarma) # store each run
    sarma_success_rates.append(success_rate) # store each run
    sarma_average_numbers.append(average_sarma)
    
    
# --- SAVE SARMA STATE ---
sarma_p_values = [d.p for d in sarma_devices]
sarma_steps = [d.step for d in sarma_devices]

np.save("sarma_p_values.npy", sarma_p_values)
np.save("sarma_steps.npy", sarma_steps)



# --- Hybrid (learning across runs) ---
hybrid_devices = [HybridDevice(i) for i in range(num_devices)]  # create Hybrid devices
hybrid_runs = []  # store all Hybrid runs
hybrid_averages = []  # store all Hybrid cumulative averages of collisions
hybrid_success_rates = []  # store all Hybrid success rates
hybrid_epsilon = []  # store all Hybrid epsilon
hybrid_average_numbers = []  # store all Hybrid average numbers of collisions


if os.path.exists("hybrid_p_values.npy"):
    saved_p = np.load("hybrid_p_values.npy")
    for d, p in zip(hybrid_devices, saved_p):
        d.p = float(p)

if os.path.exists("hybrid_epsilon.npy"):
    saved_epsilon = np.load("hybrid_epsilon.npy")
    for d, e in zip(hybrid_devices, saved_epsilon):
        d.epsilon = float(e)

for run in range(num_runs):
    t_hybrid, c_hybrid, hybrid_devices, average_collisions_hybrid, success_rate, epsilon, average_hybrid = run_simulation(
        num_devices=num_devices,
        time_units=num_steps,
        use_rl=False,
        use_sarma=False,
        use_hybrid=True,
        use_bandit=False,
        devices=hybrid_devices
    )
    hybrid_runs.append((t_hybrid, c_hybrid)) # store each run
    hybrid_averages.append(average_collisions_hybrid) # store each run
    hybrid_success_rates.append(success_rate) # store each run
    hybrid_epsilon.append(epsilon) # store each run
    hybrid_average_numbers.append(average_hybrid) # store each run
    
    
# --- SAVE HYBRID STATE ---
hybrid_p_values = [d.p for d in hybrid_devices] # collect current p for each device
hybrid_epsilon = [d.epsilon for d in hybrid_devices] # collect epsilon
hybrid_q_tables = [d.q_table for d in hybrid_devices] # collect q_table

np.save("hybrid_p_values.npy", hybrid_p_values) # save
np.save("hybrid_epsilon.npy", hybrid_epsilon) # save
np.save("hybrid_q_tables.npy", hybrid_q_tables) # save



# --- Bandit (learning across runs) ---
bandit_devices = [BanditDevice(i) for i in range(num_devices)]  # create Bandit devices
bandit_runs = []  # store all Bandit runs
bandit_averages = []  # store all Bandit cumulative averages of collisions
bandit_success_rates = []  # store all Bandit success rates
bandit_average_numbers = []  # store all Bandit average numbers of collisions


# collect current values for each device
if os.path.exists("bandit_values.npy"):
    saved_values = np.load("bandit_values.npy", allow_pickle=True)
    for d, v in zip(bandit_devices, saved_values):
        d.values = v

# collect current counts for each device
if os.path.exists("bandit_counts.npy"):
    saved_counts = np.load("bandit_counts.npy", allow_pickle=True)
    for d, c in zip(bandit_devices, saved_counts):
        d.counts = c


for run in range(num_runs):
    t_bandit, c_bandit, bandit_devices, average_collisions_bandit, success_rate, epsilon, average_bandit = run_simulation(
        num_devices=num_devices,
        time_units=num_steps,
        use_rl=False,
        use_sarma=False,
        use_hybrid=False,
        use_bandit=True,
        devices=bandit_devices
    )
    bandit_runs.append((t_bandit, c_bandit)) # store each run
    bandit_averages.append(average_collisions_bandit) # store each run
    bandit_success_rates.append(success_rate) # store each run
    bandit_average_numbers.append(average_bandit) # store each run


# --- SAVE BANDIT STATE ---
values_list = [d.values for d in bandit_devices]   # list of action value arrays
counts_list = [d.counts for d in bandit_devices]   # list of action count arrays

np.save("bandit_values.npy", values_list) # save
np.save("bandit_counts.npy", counts_list) # save



# ==============================
# 📊 PLOT 1: RL learning across runs
# ==============================
plt.figure()
for i, (t_rl, _) in enumerate(rl_runs, 1): # plot each run 
    plt.plot(t_rl, linestyle="--", label=f"RL Run {i}") # plot each run
for i, (t_base, _) in enumerate(base_runs, 1): # plot each run
    plt.plot(t_base, linestyle="-", label=f"Baseline Run {i}")
for i, (t_sarma, _) in enumerate(sarma_runs, 1): # plot each run
    plt.plot(t_sarma, linestyle=":", label=f"Sarma Run {i}")
for i, (t_hybrid, _) in enumerate(hybrid_runs, 1): # plot each run
    plt.plot(t_hybrid, linestyle=":", label=f"Hybrid Run {i}")

for i, (t_bandit, _) in enumerate(bandit_runs, 1): # plot each run
    plt.plot(t_bandit, linestyle="-.", label=f"Bandit Run {i}")

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
final_t_hybrid, final_c_hybrid = hybrid_runs[-1]
final_t_bandit, final_c_bandit = bandit_runs[-1]
plt.figure()
plt.plot(final_t_base, label="Baseline (Final Run)")
plt.plot(final_t_rl, label="RL (Final Run)")
plt.plot(final_t_sarma, label="Sarma (Final Run)")
plt.plot(final_t_hybrid, label="Hybrid (Final Run)")
plt.plot(final_t_bandit, label="Bandit (Final Run)")
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
plt.plot(final_c_hybrid, label="Hybrid (Final Run)")
plt.plot(final_c_bandit, label="Bandit (Final Run)")
plt.legend()
plt.title("Collision Comparison")
plt.xlabel("Time")
plt.ylabel("Collisions")
plt.grid(True)
plt.show()

# ==============================
# 📊 PLOT 4: Average Cumulative Collisions
# ==============================
# plt.figure()
# for i, run in enumerate(rl_averages, 1):
#     plt.plot(run, linestyle="--", label=f"RL Run {i}") # plot each run
    
# for i, run in enumerate(base_averages, 1):
#     plt.plot(run, linestyle="-", label=f"Baseline Run {i}") # plot each run

# for i, run in enumerate(sarma_averages, 1):
#     plt.plot(run, linestyle=":", label=f"Sarma Run {i}") # plot each run

# for i, run in enumerate(hybrid_averages, 1):
#     plt.plot(run, linestyle="-.", label=f"Hybrid Run {i}") # plot each run
    
# for i, run in enumerate(bandit_averages, 1):
#     plt.plot(run, linestyle="-.", label=f"Bandit Run {i}") # plot each run
# plt.legend()
# plt.title("Cumulative Average Collisions Over Time")
# plt.xlabel("Time")
# plt.ylabel("Cumulative Average Collisions Over Time Units (Each Run)")
# plt.grid(True)
# plt.show()


# # ==============================
# # Plot 5: Average Collisions
# # ==============================
# plt.figure()

# x = np.arange(1, num_runs + 1)

# plt.bar(x - 0.2, rl_average_numbers, width=0.2, label="RL")
# plt.bar(x, base_average_numbers, width=0.2, label="Baseline")
# plt.bar(x + 0.2, sarma_average_numbers, width=0.2, label="Sarma")
# plt.bar(x + 0.4, hybrid_average_numbers, width=0.2, label="Hybrid")


# plt.xlabel("Run")
# plt.ylabel("Average Collisions")
# plt.title("Average Collisions per Run")
# plt.legend()
# plt.grid(True)

# plt.show()


# # ==============================
# # 📊 PLOT 6: Success Rates
# # ==============================
# plt.figure()
# for i, run in enumerate(rl_success_rates, 1):
#     plt.plot(run, linestyle="--", label=f"RL Run {i}") # plot each run
    
# for i, run in enumerate(base_success_rates, 1):
#     plt.plot(run, linestyle="-", label=f"Baseline Run {i}") # plot each run

# for i, run in enumerate(sarma_success_rates, 1):
#     plt.plot(run, linestyle=":", label=f"Sarma Run {i}") # plot each run
    
# for i, run in enumerate(hybrid_success_rates, 1):
#     plt.plot(run, linestyle="-.", label=f"Hybrid Run {i}") # plot each run
    
# for i, run in enumerate(bandit_success_rates, 1):
#     plt.plot(run, linestyle="-.", label=f"Bandit Run {i}") # plot each run
# plt.legend()
# plt.title("Success Rate Comparison")
# plt.xlabel("Time Units")
# plt.ylabel("Cumulative Success Rates Over Time Units (Each Run)")
# plt.grid(True)
# plt.show()


# # ==============================
# # 📊 PLOT 7: Epsilon
# # ==============================
# plt.figure()
# for i, run in enumerate(rl_epsilon, 1):
#     plt.plot(run, linestyle="--", label=f"RL Run {i}") # plot each run

# plt.legend()
# plt.title("Epsilon Over Time")
# plt.xlabel("Time")
# plt.ylabel("Epsilon")
# plt.grid(True)
# plt.show()



# ==================================
# 📊 Plot 8: RL vs Baseline throughput
# ==================================
# x-axis: time
x = np.arange(1, num_steps + 1)
# y-axis: linear ALOHA line (diagonal from 0 to num_steps * 1/e)
aloha_line = np.linspace(0, num_steps * (1/np.e), num_steps)

plt.figure()

for i, (t_rl, _) in enumerate(rl_runs, 1):
    plt.plot(t_rl, linestyle="--", label=f"RL Run {i}") # plot each run

for i, (t_base, _) in enumerate(base_runs, 1):
    plt.plot(t_base, linestyle="-", label=f"Baseline Run {i}") # plot each run
    
    
plt.plot(x, aloha_line, linestyle=":", color="red", label="ALOHA Limit")


plt.legend()
plt.title("Throughput Comparison With ALOHA Limit")
plt.xlabel("Time")
plt.ylabel("Successful Transmissions")
plt.grid(True)
plt.show()




