import numpy as np
import cv2
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import rospy
import matplotlib.pyplot as plt

class ReplayBuffer():
    def __init__(self, max_size, shape, n_actions, using_camera):
        # This * 5 its cause we will generate 5 new images per image
        # using data augmentation so we have to increase our memory 
        # in order to learn with all that images
        self.mem_size = max_size * 5 if using_camera else max_size
        self.using_camera = using_camera
        self.counter = 0
        self.n_actions = n_actions

        if self.using_camera: shape = (64,64,3)

        self.states = np.zeros((self.mem_size, *shape))
        self.new_states = np.zeros((self.mem_size, *shape))

        self.actions = np.zeros((self.mem_size))

        self.rewards = np.zeros((self.mem_size))
        self.dones = np.zeros((self.mem_size))

        self.datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )

        
    def generate_data(self, batch_size):
        max_mem = min(self.counter, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.states[batch]
        new_states = self.new_states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        dones = self.dones[batch]

        return states, actions, rewards, new_states, dones


    def store_data(self, state, action, reward, new_state, done):
        
        if self.using_camera:
            self.store_data_camera(state, action, reward, new_state, done)
        else:
            index = self.counter % self.mem_size

            self.states[index] = state
            self.actions[index] = action
            self.rewards[index] = reward
            self.new_states[index] = new_state
            self.dones[index] = done

            self.counter += 1
    
    def clear_data(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.new_states = []
        self.dones = []

    def load_and_convert_image(self, image):
        bgr_image = image
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb_image = tf.image.convert_image_dtype(rgb_image, tf.float32)
        return rgb_image

    def augment_image(self, image, num_augments=5):
        #image = tf.expand_dims(image, axis=0)
        augmented_images = []
        for _ in range(num_augments):
            augmented_image = self.datagen.flow(image, batch_size=1)[0]
            augmented_images.append(tf.squeeze(augmented_image, axis=0))
        return augmented_images
        
    def store_data_camera(self, state, action, reward, new_state, done):
        # Esto lo hago debido a que cada estado va ligado a su recompensa, entonces
        # si hago data augmentation tengo que ligar las imagenes nuevas con sus probabilidades y
        # demás para que todo este ligado ALL HAVE SENSEEE BELIEVE MEEE
        num_augments = 5

        augmented_images = self.augment_image(state[np.newaxis, ...], num_augments)

        # Here we store the original image
        index = self.counter % self.mem_size

        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.new_states[index] = new_state
        self.dones[index] = done

        self.counter += 1

        for aug_img in augmented_images:
        
            """
            fig, axes = plt.subplots(1, len(augmented_images) + 1, figsize=(15, 5))
            axes[0].imshow(state * 255)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            
            for i, aug_img in enumerate(augmented_images):
                axes[i + 1].imshow(aug_img)
                axes[i + 1].set_title(f'Augmented Image {i+1}')
                axes[i + 1].axis('off')

            plt.show()
            """

            index = self.counter % self.mem_size

            self.states[index] = aug_img
            self.actions[index] = action
            self.rewards[index] = reward
            self.new_states[index] = new_state
            self.dones[index] = done

            self.counter += 1
