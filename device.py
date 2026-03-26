import random
import numpy as np

class Device:
    def __init__(self, id, p=0.5): # p = transmission probability
        self.id = id # unique identifier for the device
        self.p = random.uniform(0.1, 0.3)  # transmission probability
        self.success_streak = 0

    def choose_action(self):
        return 1 if random.random() < self.p else 0 # 1 = transmit, 0 = idle

    def update(self, result):
        # baseline adjustment
        if result == "success": # increase probability after success
            self.p = min(.9999, self.p + 0.05) # increase by 0.05 after success
            self.success_streak += 1
        elif result == "collision": # decrease probability after collision
            self.p = max(0.0001, self.p - 0.05) # decrease by 0.05 after collision
            self.success_streak = 0
        # no change after idle
        
        # fairness
        # --- Compute fairness difference ---
        # diff = my_success - avg_success

        # # --- Normalize (prevents explosion over time) ---
        # normalized_diff = diff / (avg_success + 1)

        # # --- Fairness penalty (smooth + stable) ---
        # fairness_penalty = -0.05 * normalized_diff

        # # --- Apply fairness penalty ---
        # self.p = max(0.0001, min(0.9999, self.p + fairness_penalty))

        # # --- Optional: direct behavior correction (important) ---
        # if diff > 0:  # this device is dominating
        #     self.p *= 0.95   # gently back off
        
        
        variation = random.uniform(-0.05, 0.05)  # random number between -0.05 and +0.05
        self.p = min(0.9999, max(0.0001, self.p + variation)) # add variation and ensure p stays between 0.0001 and 0.9999


# ---------------- RL VERSION ---------------- #

class RLDevice:
    def __init__(self, id): 
        self.id = id # unique identifier for the device
        self.p = random.uniform(0.1, 0.3) # transmission probability
        
        

        # Q-table: state (last result) × actions
        self.q_table = { # states: "idle", "success", "collision"
            "idle": [0, 0, 0], # 0 = decrease, 1 = same, 2 = increase
            "success": [0, 0, 0], # same actions but different states
            "collision": [0, 0, 0] 
        }
        
        
        self.success_streak = 0 # track consecutive successful transmissions
        self.last_state = "idle" # start in idle state
        self.alpha = 0.3 # learning rate
        self.gamma = 0.9 # discount factor
        self.epsilon = 0.2 # exploration rate

    def choose_action(self):
        # epsilon-greedy
        if random.random() < self.epsilon: # explore
            action = random.randint(0, 2) # 0 = decrease, 1 = same, 2 = increase
            self.epsilon = max(0.0001, self.epsilon * 0.999) # decay epsilon
        else:
            action = np.argmax(self.q_table[self.last_state]) # exploit

        # actions: 0 = decrease, 1 = same, 2 = increase
        if action == 0:  # decrease
            self.p = max(0.01, self.p-0.05)   # multiplicative decrease
        elif action == 2:  # increase
            self.p = min(0.99, self.p + 0.02)  # additive increase
        # action == 1 means keep the same probability
        
        
        self.p+=random.uniform(-0.002, 0.002) # add small random variation to encourage exploration, but only in the positive direction to prevent p from getting too low
        self.p = max(0.01, min(1.0, self.p)) # ensure p stays between 0 and 1
        return 1 if random.random() < self.p else 0, action # return both the action and the chosen action index for learning

    def update(self, state, reward, action):
        
        # --- Compute fairness difference ---
        # fairness_penalty = -0.05 * (my_success - avg_success) / max((avg_success + 1), 1)

        # # --- Combine reward ---
        # adjusted_reward = reward + fairness_penalty

        # # --- Optional: direct behavior correction (important) ---
        # if my_success>avg_success:  # this device is dominating
        #     self.p *= 0.97   # gently back off
            
        
        old_value = self.q_table[self.last_state][action] # current Q-value for the last state and action
        future = max(self.q_table[state]) # maximum Q-value for the next state

        # Q-learning update
        self.q_table[self.last_state][action] = old_value + self.alpha * ( # update the Q-value
            reward + self.gamma * future - old_value # new value = old value + learning rate * (reward + discounted future reward - old value)
        )

        self.last_state = state # update the last state to the current state for the next iteration



# ---------------- SARMA VERSION ---------------- #



class SarmaDevice:
    def __init__(self, id):
        self.id = id
        self.p = random.uniform(0.1, 0.3)
        self.step = 0.05  # adaptive step size
        self.success_streak = 0

    def choose_action(self):
        return 1 if random.random() < self.p else 0

    def update(self, result):
        # Sarma-style adaptive adjustment
        if result == "success":
            self.p += self.step*0.5 # increase success rate
            self.step = min(0.1, self.step * 1.05)  # increase confidence
            self.success_streak += 1
        elif result == "collision":
            self.p-= self.step # decrease success rate
            self.step = max(0.01, self.step * 0.9)  # reduce aggressiveness
            self.success_streak = 0
        elif result == "idle":
            self.p += self.step * 0.3 # increase success rate
            self.step = min(0.1, self.step * 1.05)
            
            
        if self.success_streak > 7:
            self.p -= self.step*0.5 # reduce success rate

        # small randomness (like your other models)
        noise = random.uniform(-self.step * 0.2, self.step * 0.2)  # proportional to step
        self.p += noise # add noise
        self.p = max(0.0001, min(0.9999, self.p))  # ensure p stays between 0.0001 and 0.9999
        


class HybridDevice:
    def __init__(self, id):
        self.id = id
        self.p = random.uniform(0.1, 0.3)

        # RL part (decision-making)
        self.q_table = {
            "idle": [0, 0, 0],
            "success": [0, 0, 0],
            "collision": [0, 0, 0]
        }

        self.last_state = "idle"
        self.alpha = 0.2
        self.gamma = 0.9
        self.epsilon = 0.2
        # Sarma part (step control)
        self.step = 0.05
        # fairness
        self.success_streak = 0

    def choose_action(self):
        # --- RL decision ---
        if random.random() < self.epsilon:
            action = random.randint(0, 2)
            self.epsilon = max(0.01, self.epsilon * 0.9995)
        else:
            action = np.argmax(self.q_table[self.last_state])

        # --- Apply Sarma-style step ---
        if action == 0:  # decrease
            self.p = max(0.01, self.p - self.step)
        elif action == 2:  # increase
            self.p = min(0.99, self.p + self.step)
        # action == 1 → no change

        # small noise (important for exploration)
        self.p += random.uniform(-0.002, 0.002)

        # clamp
        self.p = max(0.01, min(0.99, self.p))  # cap at 0.8 for fairness

        return 1 if random.random() < self.p else 0, action

    def update(self, state, reward, action):
        # --- fairness tracking ---
        # fairness_penalty = -0.05 * (my_success - avg_success) / (avg_success + 1)
        # adjusted_reward = reward + fairness_penalty
        
        # # backoff if dominating
        # if my_success > avg_success:
        #     self.p *= 0.97

        # --- RL update ---
        old = self.q_table[self.last_state][action]
        future = max(self.q_table[state])

        self.q_table[self.last_state][action] = old + self.alpha * (
            reward + self.gamma * future - old
        )

        self.last_state = state

        # --- Sarma step adaptation ---
        if state == "success":
            self.step = min(0.1, self.step * 1.05)
        elif state == "collision":
            self.step = max(0.01, self.step * 0.85)
        elif state == "idle":
            self.step = min(0.08, self.step * 1.02)
            
            
            
            
            
# ------------------- Bandit Device -------------------



class BanditDevice:
    def __init__(self, id):
        self.id = id
        self.p = random.uniform(0.1, 0.3)

        # actions: 0 = decrease, 1 = same, 2 = increase
        self.counts = [1, 1, 1]   # avoid divide-by-zero
        self.values = [0.0, 0.0, 0.0]  # average reward per action

        self.total_steps = 1

    def choose_action(self):
        # --- UCB formula ---
        ucb_values = []
        for a in range(3):
            bonus = np.sqrt(2 * np.log(self.total_steps) / self.counts[a])  
            ucb = self.values[a] + bonus 
            ucb_values.append(ucb)

        action = np.argmax(ucb_values)

        # --- apply action ---
        if action == 0:
            self.p = max(0.01, self.p * 0.9)  # gently decrease
        elif action == 2:
            self.p = min(0.99, self.p + 0.05) # gently increase

        # small noise
        self.p += random.uniform(-0.002, 0.002)

        # clamp
        self.p = max(0.001, min(0.99, self.p))

        return 1 if random.random() < self.p else 0, action

    def update(self, reward, action):
        self.total_steps += 1


        norm_reward = (reward+1.0)/2.0
        # update counts
        self.counts[action] += 1

        # incremental mean update
        n = self.counts[action]
        old = self.values[action]
        self.values[action] = old + (norm_reward - old) / n