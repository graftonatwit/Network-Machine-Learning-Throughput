import random
import numpy as np

class Device:
    def __init__(self, id, p=0.5): # p = transmission probability
        self.id = id # unique identifier for the device
        self.p = random.uniform(0.1, 0.3)  # transmission probability

    def choose_action(self):
        return 1 if random.random() < self.p else 0 # 1 = transmit, 0 = idle

    def update(self, result):
        # baseline adjustment
        if result == "success": # increase probability after success
            self.p = min(.9999, self.p + 0.05) # increase by 0.05 after success
        elif result == "collision": # decrease probability after collision
            self.p = max(0.0001, self.p - 0.05) # decrease by 0.05 after collision
        # no change after idle
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
        self.epsilon = 0.1 # exploration rate

    def choose_action(self):
        # epsilon-greedy
        if random.random() < self.epsilon: # explore
            action = random.randint(0, 2) # 0 = decrease, 1 = same, 2 = increase
            self.epsilon = max(0.0001, self.epsilon * 0.995) # decay epsilon
        else:
            action = np.argmax(self.q_table[self.last_state]) # exploit

        # actions: 0 = decrease, 1 = same, 2 = increase
        if action == 0: # decrease probability
            self.p = max(0.0, self.p - 0.05) # decrease by 0.05
        elif action == 2:
            self.p = min(1.0, self.p + 0.02) # increase by 0.02
        # action == 1 means keep the same probability
        
        self.p = min(self.p, 0.7)
        self.p+=random.uniform(-0.001, 0.001) # add small random variation to encourage exploration, but only in the positive direction to prevent p from getting too low
        self.p = max(0.001, min(1.0, self.p)) # ensure p stays between 0 and 1
        return 1 if random.random() < self.p else 0, action # return both the action and the chosen action index for learning

    def update(self, state, reward, action):
        
        if state == "success":
            self.success_streak += 1
        else:
            self.success_streak = 0
        
        fairness_penalty = 0
        if self.success_streak > 5:
            fairness_penalty = -0.5 * self.success_streak
            
        adjusted_reward = reward + fairness_penalty # adjust reward based on success streak
        old_value = self.q_table[self.last_state][action] # current Q-value for the last state and action
        future = max(self.q_table[state]) # maximum Q-value for the next state

        # Q-learning update
        self.q_table[self.last_state][action] = old_value + self.alpha * ( # update the Q-value
            adjusted_reward + self.gamma * future - old_value # new value = old value + learning rate * (reward + discounted future reward - old value)
        )

        self.last_state = state # update the last state to the current state for the next iteration
        
class SarmaDevice:
    def __init__(self, id):
        self.id = id
        self.p = random.uniform(0.1, 0.3)
        self.step = 0.05  # adaptive step size

    def choose_action(self):
        return 1 if random.random() < self.p else 0

    def update(self, result):
        # Sarma-style adaptive adjustment
        if result == "success":
            self.p += self.step*0.5 # increase success rate
            self.step = min(0.1, self.step * 1.05)  # increase confidence
        elif result == "collision":
            self.p-= self.step # decrease success rate
            self.step = max(0.01, self.step * 0.9)  # reduce aggressiveness
        elif result == "idle":
            self.p += self.step * 0.3 # increase success rate

        # small randomness (like your other models)
        noise = random.uniform(-self.step * 0.2, self.step * 0.2)  # proportional to step
        self.p += noise
        self.p = max(0.0001, min(0.9999, self.p))  # ensure p stays between 0.0001 and 0.9999