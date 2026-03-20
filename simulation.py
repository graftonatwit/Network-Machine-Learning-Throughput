from device import Device, RLDevice
import numpy as np

def run_simulation(num_devices=10, time_units=2000, use_rl=False, devices=None):
    
    if devices is None:
        if use_rl:
            devices = [RLDevice(i) for i in range(num_devices)]
        else:
            devices = [Device(i) for i in range(num_devices)]
            
        

    throughput = [] # track successful transmissions
    collisions = [] # track collisions
    
    print(f"Step 0 (initial probabilities): rl devices" if use_rl else "baseline devices")
    for i, d in enumerate(devices):
        print(f" Device {i} p = {d.p:.2f}")
    print("-" * 30)
    
    for t in range(time_units):
        actions = [] # track actions of all devices
        chosen_actions = [] # track chosen actions for RL devices

        for d in devices:
            if use_rl:
                transmit, action = d.choose_action() # get both the transmit decision and the action index for RL devices
                actions.append(transmit) # append the transmit decision (1 or 0) to actions
                chosen_actions.append(action) # append the action index to chosen_actions for learning
            else:
                actions.append(d.choose_action()) # append the transmit decision (1 or 0) to actions

        total = sum(actions) # total number of devices transmitting

        if total == 1:
            result = "success" # successful transmission
            reward = 1 # reward for success
        elif total > 1:
            result = "collision" # collision occurred
            reward = -total # penalty for collision
        else:
            result = "idle" # no transmission
            reward = 0 # no reward for idle
            
        if use_rl:
            avg_p = sum(d.p for d in devices) / len(devices)
            reward -= 0.1 * avg_p
        # update devices
        for i, d in enumerate(devices): # loop through devices to update them based on the result
            if use_rl: # if using RL devices, we need to provide the state and reward for learning
                d.update(result, reward, chosen_actions[i]) # update the device with the result, reward, and the action it took
            else:
                d.update(result) # update the device with the result for baseline devices
                
        throughput.append(1 if result == "success" else 0) # append 1 for success, 0 otherwise
        collisions.append(1 if result == "collision" else 0) # append 1 for collision, 0 otherwise

        # --- print probabilities every 500 steps ---
        if (t + 1) % 500 == 0:
            print(f"Step {t+1}: {('rl devices' if use_rl else 'baseline devices')}")
            for i, d in enumerate(devices):
                action_str = "transmitted" if actions[i] == 1 else "idle"
                print(f" Device {i} p = {d.p:.4f} ({action_str})")
            print("-" * 30)
    return np.cumsum(throughput), np.cumsum(collisions), devices  # return cumulative throughput and collisions over time