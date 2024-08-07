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
        input_shape = (64, 64, 3)
        """
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
        """

        self.conv_block1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(3, (3,3), strides=(1,1), padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        ])

        self.conv_block2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, (3,3), strides=(1,1), padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        ])

        self.dense_net = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(128),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(n_actions)
        ])

        
        """
        self.conv1 = tf.keras.layers.Conv2D(conv1_dims[0], conv1_dims[1], activation='relu', padding='same', input_shape=input_shape)
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(conv2_dims[0], conv2_dims[1], activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(fc1_dims, activation='relu')
        self.fc2 = tf.keras.layers.Dense(n_actions, activation='softmax')
        """

    def call(self, state):
        """
        x = self.conv1(state)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        """
        x = self.conv_block1(state)
        x = self.conv_block2(x)
        x = self.dense_net(x)
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

        input_shape = (64, 64, 3)
        """
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
       

        self.conv1 = tf.keras.layers.Conv2D(conv1_dims[0], conv1_dims[1], activation='relu', padding='same', input_shape=input_shape)
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(conv2_dims[0], conv2_dims[1], activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(fc1_dims, activation='relu')
        self.fc2 = tf.keras.layers.Dense(fc2_dims, activation='relu')
        self.q_value = tf.keras.layers.Dense(n_actions)
        """

        self.conv_block1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(3, (3,3), strides=(1,1), padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        ])

        self.conv_block2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, (3,3), strides=(1,1), padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        ])

        self.dense_net = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(128),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(n_actions)
        ])

    def call(self, state):
        """
        x = self.conv1(state)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.q_value(x)
        """
        x = self.conv_block1(state)
        x = self.conv_block2(x)
        x = self.dense_net(x)
        return x

    #def call(self, state):
        #return self.net(state)
    