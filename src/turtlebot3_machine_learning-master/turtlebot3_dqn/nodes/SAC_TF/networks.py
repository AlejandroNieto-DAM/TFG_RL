import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

class CustomInitializer(tf.keras.initializers.Initializer):
    def __init__(self):
        pass

    def __call__(self, shape, dtype=None):
        fan_in = shape[0]
        lim = 1. / np.sqrt(fan_in)
        return tf.random.uniform(shape, -lim, lim, dtype=dtype)

    def get_config(self):
        return {}


class Actor(tf.keras.Model):
    def __init__(self, input_dims, n_actions, fc1_dims, fc2_dims, name, save_directory='model_weights/sac/'):
        super(Actor, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.model_name = name

        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(fc1_dims, input_shape=(input_dims[0],), kernel_initializer=CustomInitializer()),
            tf.keras.layers.PReLU(),
            tf.keras.layers.Dense(fc2_dims,  kernel_initializer=CustomInitializer()),
            tf.keras.layers.PReLU(),
            tf.keras.layers.Dense(n_actions,  kernel_initializer=CustomInitializer())
        ])

    def call(self, state):
        x = self.net(state)
        x = tf.nn.softmax(x, axis=-1)
        return x

    def evaluate(self, state, epsilon=1e-8):
        action_probs = self.call(state)
        dist = tfp.distributions.Categorical(probs=action_probs)
        action = dist.sample()
        
        # Handling 0.0 probabilities
        z = tf.cast(action_probs == 0.0, dtype=tf.float32) * epsilon
        log_action_probabilities = tf.math.log(action_probs + z)
        
        return tf.convert_to_tensor(action), tf.convert_to_tensor(action_probs), tf.convert_to_tensor(log_action_probabilities)

    def get_det_action(self, state):
        action_probs = self.call(state)
        dist = tfp.distributions.Categorical(probs=action_probs)
        action = dist.sample()
        return action

class Critic(tf.keras.Model):
    def __init__(self, input_dims, n_actions, fc1_dims, fc2_dims, name, save_directory='model_weights/sac/'):
        super(Critic, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.model_name = name

        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(fc1_dims, input_shape=(input_dims[0],), kernel_initializer=CustomInitializer()),
            tf.keras.layers.PReLU(),
            tf.keras.layers.Dense(fc2_dims, kernel_initializer=CustomInitializer()),
            tf.keras.layers.PReLU(),
            tf.keras.layers.Dense(n_actions, kernel_initializer=CustomInitializer())
        ])

    def call(self, state):
        return self.net(state)



class XavierInitializer(tf.keras.initializers.Initializer):
    def __init__(self):
        pass

    def __call__(self, shape, dtype=None):
        fan_in = shape[0]
        fan_out = shape[1]
        limit = np.sqrt(6 / (fan_in + fan_out))
        return tf.random.uniform(shape, -limit, limit, dtype=dtype)

    def get_config(self):
        return {}
    
class CNNActor(tf.keras.Model):
    def __init__(self, conv1_dims, conv2_dims, n_actions, fc1_dims, fc2_dims, name, save_directory='model_weights/sac/'):
        super(CNNActor, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.model_name = name

        self.net = tf.keras.Sequential([
            tf.keras.layers.Conv2D(conv1_dims[0], conv1_dims[1], activation='relu', padding='same', kernel_initializer=XavierInitializer()),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(conv2_dims[0], conv2_dims[1], activation='relu', padding='same', kernel_initializer=XavierInitializer()),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(fc1_dims, kernel_initializer=XavierInitializer()),
            tf.keras.layers.PReLU(),
            tf.keras.layers.Dense(fc2_dims,  kernel_initializer=XavierInitializer()),
            tf.keras.layers.PReLU(),
            tf.keras.layers.Dense(n_actions,  kernel_initializer=XavierInitializer())
        ])

    def call(self, state):
        x = self.net(state)
        x = tf.nn.softmax(x, axis=-1)
        return x

    def evaluate(self, state, epsilon=1e-8):
        action_probs = self.call(state)
        dist = tfp.distributions.Categorical(probs=action_probs)
        action = dist.sample()
        
        # Handling 0.0 probabilities
        z = tf.cast(action_probs == 0.0, dtype=tf.float32) * epsilon
        log_action_probabilities = tf.math.log(action_probs + z)
        
        return tf.convert_to_tensor(action), tf.convert_to_tensor(action_probs), tf.convert_to_tensor(log_action_probabilities)

    def get_det_action(self, state):
        action_probs = self.call(state)
        dist = tfp.distributions.Categorical(probs=action_probs)
        action = dist.sample()
        return action

class CNNCritic(tf.keras.Model):
    def __init__(self, conv1_dims, conv2_dims, n_actions, fc1_dims, fc2_dims, name, save_directory='model_weights/sac/'):
        super(CNNCritic, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.model_name = name

        self.net = tf.keras.Sequential([
            tf.keras.layers.Conv2D(conv1_dims[0], conv1_dims[1], activation='relu', padding='same', kernel_initializer=XavierInitializer()),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(conv2_dims[0], conv2_dims[1], activation='relu', padding='same', kernel_initializer=XavierInitializer()),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(fc1_dims, kernel_initializer=XavierInitializer()),
            tf.keras.layers.PReLU(),
            tf.keras.layers.Dense(fc2_dims,  kernel_initializer=XavierInitializer()),
            tf.keras.layers.PReLU(),
            tf.keras.layers.Dense(n_actions,  kernel_initializer=XavierInitializer())
        ])
        
    def call(self, state):
        return self.net(state)
    