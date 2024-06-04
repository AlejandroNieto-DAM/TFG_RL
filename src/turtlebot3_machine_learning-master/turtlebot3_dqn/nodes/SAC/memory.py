import numpy as np

class Memory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.rewards = []
        self.new_states = []
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

        return np.array(self.states), np.array(self.actions), np.array(self.rewards), np.array(self.new_states), np.array(self.dones), indices_batches

    def store_data(self, states, actions, rewards, new_state, done):
        self.states.append(states)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.new_states.append(new_state)
        self.dones.append(done)
    
    def clear_data(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.new_states = []
        self.dones = []