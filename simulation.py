from device import Device, RLDevice, SarmaDevice, HybridDevice, BanditDevice
import numpy as np

def run_simulation(num_devices=10, time_units=2000, use_rl=False, use_sarma=False, use_hybrid=False,  use_bandit=False, devices=None):
    
    if devices is None:
        if use_rl:
            devices = [RLDevice(i) for i in range(num_devices)] # create RL devices
        elif use_sarma:
            devices = [SarmaDevice(i) for i in range(num_devices)] # create SARMA devices
        elif use_hybrid:
            devices = [HybridDevice(i) for i in range(num_devices)] # create hybrid devices
        elif use_bandit:
            devices = [BanditDevice(i) for i in range(num_devices)] # create bandit devices
        else:
            devices = [Device(i) for i in range(num_devices)] # create baseline devices
            
        
    average_collisions_list = [] # track average collisions
    throughput = [] # track successful transmissions
    collisions = [] # track collisions
    epsilon_history = [] # track epsilon
    success_counts = [0] * len(devices) # track number of successful transmissions
    
    
    
    
    
    for t in range(time_units):
        actions = [] # track actions of all devices
        chosen_actions = [] # track chosen actions for RL devices
        individual_states = [] # track actions of all devices

        for i, d in enumerate(devices):
            if use_rl:
                transmit, action = d.choose_action() # get both the transmit decision and the action index for RL devices
                actions.append(transmit) # append the transmit decision (1 or 0) to actions
                chosen_actions.append(action) # append the action index to chosen_actions for learning
                
            elif use_sarma:
                actions.append(d.choose_action()) # append the transmit decision (1 or 0) to actions
                
            elif use_hybrid:
                transmit, action = d.choose_action() # get both the transmit decision and the action index for hybrid devices
                actions.append(transmit) # append the transmit decision (1 or 0) to actions
                chosen_actions.append(action) # append the action index to chosen_actions for learning
                
            elif use_bandit:
                transmit, action = d.choose_action() # get both the transmit decision and the action index for hybrid devices
                actions.append(transmit) # append the transmit decision (1 or 0) to actions
                chosen_actions.append(action) # append the action index to chosen_actions for learning

            else:
                actions.append(d.choose_action()) # append the transmit decision (1 or 0) to actions

                
        
        if t == 0:
            if use_rl:
                print(f"Step 1 (initial probabilities) rl devices ")
            elif use_sarma:
                print(f"Step 1 (initial probabilities) sarma devices ")
            elif use_hybrid:
                print(f"Step 1 (initial probabilities) hybrid devices ")
            elif use_bandit:
                print(f"Step 1 (initial probabilities) bandit devices ")
            else:
                print(f"Step 1 (initial probabilities) baseline devices ")
            
            for i, d in enumerate(devices):
                action_str = "transmitted" if actions[i] == 1 else "idle"
                print(f" Device {i+1} p = {d.p:.4f} ({action_str})")
                
            print("-" * 30)
        
        
        
        total = sum(actions) # total number of devices transmitting



        if total == 1:
            result = "success" # successful transmission
            avg_p = sum(d.p for d in devices) / len(devices)
            global_reward = 1 # reward for success
            collisions_this_step = 0 # track average collisions
            # winner = actions.index(1) # index of the winner
            # success_counts[winner] += 1 # track number of successful transmissions
        elif total > 1:
            result = "collision" # collision occurred
            avg_p = sum(d.p for d in devices) / len(devices)
            global_reward = -avg_p*total
            collisions_this_step = total # track average collisions
            
        else:
            result = "idle" # no transmission
            global_reward = -.02 # no reward for idle
            collisions_this_step = 0 # track average collisions
        
        
        # for i, d in enumerate(devices):
        #     if result == "success":
        #         if i == winner:
        #             individual_states.append("success")
        #         else:
        #             individual_states.append("idle")

        #     elif result == "collision":
        #         if actions[i] == 1:
        #             individual_states.append("collision")
        #         else:
        #             individual_states.append("idle")

        #     else:
        #         individual_states.append("idle") # idle state for baseline devices
        
        
        average_collisions_list.append(collisions_this_step) # track average collisions
        
        avg_success = sum(success_counts) / len(devices) # calculate average success
        
        
        
        # for i, d in enumerate(devices):
        #     state = individual_states[i]

        #     if state == "success":
        #         individual_reward = 1
        #     elif state == "collision":
        #         individual_reward = -0.5
        #     else:
        #         individual_reward = -0.02

        #     reward = global_reward + individual_reward

        #     if use_rl:
        #         d.update(state, reward, chosen_actions[i], success_counts[i], avg_success)
        #     elif use_hybrid:
        #         d.update(state, reward, chosen_actions[i], success_counts[i], avg_success)
        #     elif use_sarma:
        #         d.update(result)
        #     elif use_bandit:
        #         d.update(reward, chosen_actions[i])
        #     else:
        #         d.update(state, success_counts[i], avg_success) # update the device with the result for baseline devices
        
        

        
        
                
        throughput.append(1 if result == "success" else 0) # append 1 for success, 0 otherwise
        collisions.append(1 if result == "collision" else 0) # append 1 for collision, 0 otherwise
        
        # track epsilon
        if use_rl or use_hybrid:
            avg_epsilon = np.mean([d.epsilon for d in devices])  # average over all RL devices
            epsilon_history.append(avg_epsilon)
                


        # update probabilities for all devices
        for i, d in enumerate(devices):
            if use_sarma:
                d.update(result)
            elif use_rl:
                d.update(result, global_reward, chosen_actions[i])
            elif use_hybrid:
                d.update(result, global_reward, chosen_actions[i])
            elif use_bandit:
                d.update(global_reward, chosen_actions[i])
            else:
                d.update(result)

        # --- print probabilities every 1000 steps ---
        if (t + 1) % 250 == 0:
            
            if use_rl:
                print(f"Step {t+1}  rl devices ")
            elif use_sarma:
                print(f"Step {t+1}  sarma devices ")
            elif use_hybrid:
                print(f"Step {t+1}  hybrid devices ")
            elif use_bandit:
                print(f"Step {t+1}  bandit devices ")
            else:
                print(f"Step {t+1}  baseline devices ")
            
            for i, d in enumerate(devices):
                action_str = "transmitted" if actions[i] == 1 else "idle"
                if use_rl or use_hybrid:
                    print(f" Device {i+1} p = {d.p:.4f} ({action_str})")
                else:
                    print(f" Device {i+1} p = {d.p:.4f} ({action_str})")
            print("-" * 30)
                
            
        
    # calculate average collisions
    average_collisions = sum(average_collisions_list) / len(average_collisions_list)
    
    # calculate cumulative average collisions    
    cumulative_avg_collisions = np.cumsum(average_collisions_list) / (np.arange(1, time_units+1))
    # calculate cumulative success
    cumulative_success = np.cumsum(throughput) / np.arange(1, len(throughput)+1)
    print(f"Average collisions: {cumulative_avg_collisions[-1]:.4f}") # print average collisions
    if use_rl or use_hybrid:
        print(f"Average epsilon (final run): {epsilon_history[-1]:.4f}")
    print("-" * 30)

    return np.cumsum(throughput), np.cumsum(collisions), devices, cumulative_avg_collisions, cumulative_success, epsilon_history, average_collisions  # return cumulative throughput and collisions and devices and average collisions