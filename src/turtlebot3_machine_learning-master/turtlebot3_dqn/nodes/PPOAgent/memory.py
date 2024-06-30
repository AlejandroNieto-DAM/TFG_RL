import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
import tensorflow as tf
import cv2

class Memory:
    def __init__(self, batch_size, using_camera):
        self.using_camera = using_camera
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

    def store_data(self, state, prob, val, action, reward, done):

        # Hay que testar esto
        if self.using_camera:
            self.store_data_camera(state, prob, val, action, reward, done)
        else:
            self.states.append(state)
            self.probs.append(prob)
            self.vals.append(val)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)
    
    def clear_data(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

    
    def load_and_convert_image(self, image):
        bgr_image = image
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb_image = tf.image.convert_image_dtype(rgb_image, tf.float32)
        return rgb_image

    def augment_image(self, image, num_augments=5):
        
        # Debo mirar algunos de estos parámetros porque no se que hacen?
        # Paula tiene algo de noise pero no aparece como un parámetro tengo que mirarrrr
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )

        image = tf.expand_dims(image, axis=0)
        augmented_images = []
        for _ in range(num_augments):
            augmented_image = datagen.flow(image, batch_size=1)[0]
            augmented_images.append(tf.squeeze(augmented_image, axis=0))
        return augmented_images
        
    def store_data_camera(self, state, prob, val, action, reward, done):
        
        # Esto lo hago debido a que cada estado va ligado a su recompensa, entonces
        # si hago data augmentation tengo que ligar las imagenes nuevas con sus probabilidades y
        # demás para que todo este ligado ALL HAVE SENSEEE BELIEVE MEEE
        #processed_image = load_and_convert_image(image_path)
        num_augments = 5
        augmented_images = self.augment_image(state, num_augments)

        # Here we save the original image
        self.states.append(state)
        self.probs.append(prob)
        self.vals.append(val)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

        """
        for aug_img in augmented_images:
            self.states.append(aug_img)
            self.probs.append(prob)
            self.vals.append(val)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)
        """


