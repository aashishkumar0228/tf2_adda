import tensorflow as tf

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
from tensorflow.keras import layers
import time

from models import *

cross_entropy = tf.keras.losses.BinaryCrossentropy()

class ADDA():
    def __init__(self, sim_data_path, real_data_path, source_encoder_model_path, epochs=10, batch_size=256):
        self.num_of_classes = 35
        self.source_encoder_model_path = source_encoder_model_path
        self.epochs = epochs
        self.batch_size = batch_size

        # load simulated data
        self.df_sim = pd.read_csv(sim_data_path, header = None)
        self.total_sim_samples = self.df_sim.shape[0]
        self.x_sim = np.asarray(self.df_sim.iloc[:self.total_sim_samples,1:]).reshape([self.total_sim_samples,28,28,1]) # taking all columns expect column 0
        self.x_sim = self.x_sim / 255

        # load real data
        self.df_real = pd.read_csv(real_data_path, header = None)
        self.total_real_samples = self.df_real.shape[0]
        self.x_real = np.asarray(self.df_real.iloc[:self.total_real_samples,1:]).reshape([self.total_real_samples,28,28,1]) # taking all columns expect column 0
        self.x_real = self.x_real / 255


        self.source_encoder = tf.keras.models.load_model(self.source_encoder_model_path)
        print('Source encoder summary')
        print(self.source_encoder.summary())

        self.target_encoder = get_source_encoder()
        self.target_encoder.set_weights(self.source_encoder.get_weights())

        for layer in self.source_encoder.layers:
            layer.trainable = False
        
        self.discriminator = get_discriminator()

        self.target_encoder_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    
    def discriminator_loss(self, sim_output, real_output):
        sim_loss = cross_entropy(tf.ones_like(sim_output), sim_output)
        real_loss = cross_entropy(tf.zeros_like(real_output), real_output)
        total_loss = sim_loss + real_loss
        return total_loss

    def target_encoder_loss(self, real_output):
        return cross_entropy(tf.ones_like(real_output), real_output)

    def train(self):
        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(target_encoder_optimizer=self.target_encoder_optimizer,
                                        discriminator_optimizer=self.discriminator_optimizer,
                                        target_encoder=self.target_encoder,
                                        discriminator=self.discriminator)
        

        for epoch in range(self.epochs):
            print("Epoch: ", epoch)
            num_steps = int(self.total_real_samples / self.batch_size)
            np.random.shuffle(self.x_sim)
            np.random.shuffle(self.x_real)
            for idx in range(num_steps):
                idx_start = idx * self.batch_size
                idx_end = min((idx + 1) * self.batch_size, self.total_real_samples)
                x_sim_batch = self.x_sim[idx_start : idx_end]
                x_real_batch = self.x_real[idx_start : idx_end]
                self.train_step(x_sim_batch, x_real_batch)
            
            if (epoch + 1) % 5 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
        
        checkpoint.save(file_prefix = checkpoint_prefix)
        self.target_encoder.save_weights("target_encoder_weights.h5")
        self.target_encoder.save('target_encoder_model')
                
    @tf.function
    def train_step(self, x_sim_batch, x_real_batch):
        with tf.GradientTape() as enc_tape, tf.GradientTape() as disc_tape:
            sim_feature = self.source_encoder(x_sim_batch, training=False)
            real_feature = self.target_encoder(x_real_batch, training=True)

            sim_output = self.discriminator(sim_feature, training=True)
            real_output = self.discriminator(real_feature, training=True)

            target_encoder_loss = self.target_encoder_loss(real_output)
            disc_loss = self.discriminator_loss(sim_output, real_output)
        
        gradients_of_target_encoder = enc_tape.gradient(target_encoder_loss, self.target_encoder.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.target_encoder_optimizer.apply_gradients(zip(gradients_of_target_encoder, self.target_encoder.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        


if __name__ == '__main__':
    sim_data_path = 'emnist_combined_uppercase_lowercase_rotation_corrected_inverted_background_train.csv'
    real_data_path = 'handwriting_uppercase_lowercase_aashish_kids_phase1_background_mirror_18th_jan_augmented_new_train.csv'
    source_encoder_model_path = "source_encoder_model"

    epochs = 10
    batch_size = 256

    adda_1 = ADDA(sim_data_path=sim_data_path,
                  real_data_path=real_data_path,
                  source_encoder_model_path=source_encoder_model_path,
                  epochs=epochs,
                  batch_size=batch_size)
    
    adda_1.train()

    
