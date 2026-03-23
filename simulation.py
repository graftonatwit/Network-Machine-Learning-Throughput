from device import Device, RLDevice, SarmaDevice
import numpy as np

def run_simulation(num_devices=10, time_units=2000, use_rl=False, use_sarma=False, devices=None):
    
    if devices is None:
        if use_rl:
            devices = [RLDevice(i) for i in range(num_devices)] # create RL devices
        elif use_sarma:
            devices = [SarmaDevice(i) for i in range(num_devices)]
        else:
            devices = [Device(i) for i in range(num_devices)]
            
        
    average_collisions_list = []
    throughput = [] # track successful transmissions
    collisions = [] # track collisions
    epsilon_history = [] # track epsilon
    latency_check = [] # track latency
    
    
    
    for t in range(time_units):
        actions = [] # track actions of all devices
        chosen_actions = [] # track chosen actions for RL devices

        for d in devices:
            if use_rl:
                transmit, action = d.choose_action() # get both the transmit decision and the action index for RL devices
                actions.append(transmit) # append the transmit decision (1 or 0) to actions
                chosen_actions.append(action) # append the action index to chosen_actions for learning
                
            elif use_sarma:
                actions.append(d.choose_action())
            else:
                actions.append(d.choose_action()) # append the transmit decision (1 or 0) to actions
                
        
        if t == 0:
            if use_rl:
                print(f"Step 1 (initial probabilities) rl devices ")
            elif use_sarma:
                print(f"Step 1 (initial probabilities) sarma devices ")
            else:
                print(f"Step 1 (initial probabilities) baseline devices ")
            
            for i, d in enumerate(devices):
                action_str = "transmitted" if actions[i] == 1 else "idle"
                print(f" Device {i+1} p = {d.p:.4f} ({action_str})")
                
            print("-" * 30)
        
        total = sum(actions) # total number of devices transmitting



        if total == 1:
            result = "success" # successful transmission
            reward = 1 # reward for success
            collisions_this_step = 0
        elif total > 1:
            result = "collision" # collision occurred
            avg_p = sum(d.p for d in devices) / len(devices)
            reward = -avg_p # reward for collision
            collisions_this_step = total # track average collisions
            
        else:
            result = "idle" # no transmission
            reward = 0 # no reward for idle
            collisions_this_step = 0 # track average collisions
        
        
        
        average_collisions_list.append(collisions_this_step) # track average collisions
        
        
        
        # update devices
        for i, d in enumerate(devices): # loop through devices to update them based on the result
            if use_rl: # if using RL devices, we need to provide the state and reward for learning
                d.update(result, reward, chosen_actions[i]) # update the device with the result, reward, and the action it took
            
            elif use_sarma:
                d.update(result)
            else:
                d.update(result) # update the device with the result for baseline devices
                
        throughput.append(1 if result == "success" else 0) # append 1 for success, 0 otherwise
        collisions.append(1 if result == "collision" else 0) # append 1 for collision, 0 otherwise
        
        # track epsilon
        if use_rl:
            avg_epsilon = np.mean([d.epsilon for d in devices])  # average over all RL devices
            epsilon_history.append(avg_epsilon)
                


        # --- print probabilities every 1000 steps ---
        if (t + 1) % 250 == 0:
            
            if use_rl:
                print(f"Step {t+1}  rl devices ")
            elif use_sarma:
                print(f"Step {t+1}  sarma devices ")
            else:
                print(f"Step {t+1}  baseline devices ")
            
            for i, d in enumerate(devices):
                action_str = "transmitted" if actions[i] == 1 else "idle"
                if use_rl:
                    print(f" Device {i+1} p = {d.p:.4f} ({action_str})")
                else:
                    print(f" Device {i+1} p = {d.p:.4f} ({action_str})")
            print("-" * 30)
                
            
            
    
    # calculate average collisions    
    cumulative_avg_collisions = np.cumsum(average_collisions_list) / (np.arange(1, time_units+1))
    cumulative_success = np.cumsum(throughput) / np.arange(1, len(throughput)+1)
    print(f"Average collisions: {cumulative_avg_collisions[-1]:.4f}") # print average collisions
    if use_rl:
        print(f"Average epsilon (final run): {epsilon_history[-1]:.4f}")
    print("-" * 30)

    return np.cumsum(throughput), np.cumsum(collisions), devices, cumulative_avg_collisions, cumulative_success, epsilon_history  # return cumulative throughput and collisions and devices and average collisions