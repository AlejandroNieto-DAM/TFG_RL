import numpy as np

class Memory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_data(self):
        indices_muestras = np.arange(len(self.states))
        np.random.shuffle(indices_muestras)
        num_batches = int(len(self.states) / self.batch_size)
        indices_batches = []
        
        for i_batch in range(num_batches):
            indices_batches.append(indices_muestras[self.batch_size * i_batch : self.batch_size * i_batch + self.batch_size])

        if len(self.states) % self.batch_size != 0:     
            indices_batches.append(indices_muestras[self.batch_size * (num_batches) : ])

        return np.array(self.states), np.array(self.probs), np.array(self.vals), np.array(self.actions), np.array(self.rewards), np.array(self.dones), indices_batches

    def store_data(self, states, probs, vals, actions, rewards, dones):
        self.states.append(states)
        self.probs.append(probs)
        self.vals.append(vals)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.dones.append(dones)
    
    def clear_data(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []