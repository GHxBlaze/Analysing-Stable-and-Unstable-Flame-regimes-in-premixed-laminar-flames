import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, List, Dict, Optional
import glob


class FlameDataProcessor:
    
    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.data_files = []
        self.parameters = []
        self.sequences = []
        
    def extract_parameters_from_filename(self, filename: str) -> Tuple[float, float]:
        phi_match = re.search(r'Phi_(\d+)p?(\d*)', filename)
        u_match = re.search(r'u_(\d+)p?(\d*)', filename)
        
        if phi_match:
            phi_int = phi_match.group(1)
            phi_dec = phi_match.group(2) if phi_match.group(2) else '0'
            phi = float(f"{phi_int}.{phi_dec}")
        else:
            raise ValueError(f"Cannot extract phi from filename: {filename}")
            
        if u_match:
            u_int = u_match.group(1)
            u_dec = u_match.group(2) if u_match.group(2) else '0'
            u = float(f"{u_int}.{u_dec}")
        else:
            raise ValueError(f"Cannot extract u from filename: {filename}")
            
        return phi, u
    
    def load_data_file(self, filepath: str) -> np.ndarray:
        try:
            data = pd.read_csv(filepath, sep='\t', header=None, 
                                         names=['heat', 'time', 'pressure'])
            return data.values.astype(np.float32)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def load_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        txt_files = glob.glob(os.path.join(self.data_dir, "*.txt"))
        
        sequences = []
        parameters = []
        
        for filepath in txt_files:
            filename = os.path.basename(filepath)
            try:
                phi, u = self.extract_parameters_from_filename(filename)
                data = self.load_data_file(filepath)
                
                if data is not None:
                    if len(data) > 500:
                        data = data[:500]  # Truncate
                elif len(data) < 500:
                    # Pad shorter sequences with zeros or last value
                        padding = np.tile(data[-1], (500 - len(data), 1))
                        data = np.vstack([data, padding])
                        
                sequences.append(data)
                parameters.append([phi, u])
                print(f"Loaded {filename}: phi={phi}, u={u}, shape={data.shape}")
                    
            except Exception as e:
                print(f"Skipping {filename}: {e}")
        
        if sequences:
            min_len = min(seq.shape[0] for seq in sequences)
            print(f"Truncating all sequences to length: {min_len}")
            sequences = [seq[:5000:10] for seq in sequences]
        
        return np.array(sequences), np.array(parameters)
    
    def normalize_data(self, sequences: np.ndarray) -> Tuple[np.ndarray, Dict]:
        global_min = np.min(sequences)
        global_max = np.max(sequences)
        
        normalized = (sequences - global_min) / (global_max - global_min)
        
        norm_params = {
            'global_min': global_min,
            'global_max': global_max
        }
        
        return normalized, norm_params
    
    def denormalize_data(self, normalized_sequences: np.ndarray, 
                        norm_params: Dict) -> np.ndarray:
        return (normalized_sequences * (norm_params['global_max'] - norm_params['global_min']) + 
                norm_params['global_min'])


class TimeGAN:
    
    def __init__(self, seq_len: int, n_features: int, n_conditions: int = 2,
                 hidden_dim: int = 24, n_layers: int = 3):
        self.seq_len = seq_len
        self.n_features = n_features
        self.n_conditions = n_conditions
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedder = self.build_embedder()
        self.recovery = self.build_recovery()
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.supervisor = self.build_supervisor()
        
        self.optimizer_g = keras.optimizers.Adam(learning_rate=0.001)
        self.optimizer_d = keras.optimizers.Adam(learning_rate=0.001)
        self.optimizer_e = keras.optimizers.Adam(learning_rate=0.001)
        
    def build_embedder(self) -> keras.Model:
        inputs = layers.Input(shape=(self.seq_len, self.n_features), name='real_data')
        conditions = layers.Input(shape=(self.n_conditions,), name='conditions')
        
        cond_repeated = layers.RepeatVector(self.seq_len)(conditions)
        
        combined = layers.Concatenate(axis=-1)([inputs, cond_repeated])
        
        x = combined
        for i in range(self.n_layers):
            x = layers.LSTM(self.hidden_dim, return_sequences=True, 
                          name=f'embedder_lstm_{i}')(x)
        
        outputs = layers.Dense(self.hidden_dim, activation='sigmoid', 
                             name='embedder_output')(x)
        
        return keras.Model([inputs, conditions], outputs, name='embedder')
    
    def build_recovery(self) -> keras.Model:
        inputs = layers.Input(shape=(self.seq_len, self.hidden_dim), name='hidden_states')
        
        x = inputs
        for i in range(self.n_layers):
            x = layers.LSTM(self.hidden_dim, return_sequences=True,
                          name=f'recovery_lstm_{i}')(x)
        
        outputs = layers.Dense(self.n_features, name='recovery_output')(x)
        
        return keras.Model(inputs, outputs, name='recovery')
    
    def build_generator(self) -> keras.Model:
        noise = layers.Input(shape=(self.seq_len, self.n_features), name='noise')
        conditions = layers.Input(shape=(self.n_conditions,), name='conditions')
        
        cond_repeated = layers.RepeatVector(self.seq_len)(conditions)
        
        combined = layers.Concatenate(axis=-1)([noise, cond_repeated])
        
        x = combined
        for i in range(self.n_layers):
            x = layers.LSTM(self.hidden_dim, return_sequences=True,
                          name=f'generator_lstm_{i}')(x)
        
        outputs = layers.Dense(self.hidden_dim, activation='sigmoid',
                             name='generator_output')(x)
        
        return keras.Model([noise, conditions], outputs, name='generator')
    
    def build_discriminator(self) -> keras.Model:
        inputs = layers.Input(shape=(self.seq_len, self.hidden_dim), name='hidden_states')
        conditions = layers.Input(shape=(self.n_conditions,), name='conditions')
        
        cond_repeated = layers.RepeatVector(self.seq_len)(conditions)
        
        combined = layers.Concatenate(axis=-1)([inputs, cond_repeated])
        
        x = combined
        for i in range(self.n_layers):
            x = layers.LSTM(self.hidden_dim, return_sequences=True,
                          name=f'discriminator_lstm_{i}')(x)
        
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(1, activation='sigmoid', name='discriminator_output')(x)
        
        return keras.Model([inputs, conditions], outputs, name='discriminator')
    
    def build_supervisor(self) -> keras.Model:
        inputs = layers.Input(shape=(self.seq_len-1, self.hidden_dim), name='hidden_states')
        
        x = inputs
        for i in range(self.n_layers-1):
            x = layers.LSTM(self.hidden_dim, return_sequences=True,
                          name=f'supervisor_lstm_{i}')(x)
        
        outputs = layers.Dense(self.hidden_dim, activation='sigmoid',
                             name='supervisor_output')(x)
        
        return keras.Model(inputs, outputs, name='supervisor')
    
    @tf.function
    def train_step(self, real_data: tf.Tensor, conditions: tf.Tensor) -> Dict:
        batch_size = tf.shape(real_data)[0]
        
        noise = tf.random.normal((batch_size, self.seq_len, self.n_features))
        
        with tf.GradientTape() as tape:
            h_real = self.embedder([real_data, conditions])
            x_reconstructed = self.recovery(h_real)
            
            e_loss = tf.reduce_mean(tf.square(real_data - x_reconstructed))
        
        e_gradients = tape.gradient(e_loss, 
                                   self.embedder.trainable_variables + 
                                   self.recovery.trainable_variables)
        self.optimizer_e.apply_gradients(zip(e_gradients, 
                                           self.embedder.trainable_variables + 
                                           self.recovery.trainable_variables))
        
        with tf.GradientTape() as tape:
            h_fake = self.generator([noise, conditions])
            x_fake = self.recovery(h_fake)
            
            d_fake = self.discriminator([h_fake, conditions])
            
            g_loss_adv = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.ones_like(d_fake), d_fake))
            
            g_loss = g_loss_adv
        
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.optimizer_g.apply_gradients(zip(g_gradients, 
                                           self.generator.trainable_variables))
        
        with tf.GradientTape() as tape:
            h_real = self.embedder([real_data, conditions])
            h_fake = self.generator([noise, conditions])
            
            d_real = self.discriminator([h_real, conditions])
            d_fake = self.discriminator([h_fake, conditions])
            
            d_loss_real = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.ones_like(d_real), d_real))
            d_loss_fake = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.zeros_like(d_fake), d_fake))
            d_loss = d_loss_real + d_loss_fake
        
        d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.optimizer_d.apply_gradients(zip(d_gradients, 
                                           self.discriminator.trainable_variables))
        
        return {
            'e_loss': e_loss,
            'g_loss': g_loss,
            'd_loss': d_loss,
            'd_real': tf.reduce_mean(d_real),
            'd_fake': tf.reduce_mean(d_fake)
        }
    
    def generate_samples(self, conditions: np.ndarray, n_samples: int) -> np.ndarray:
        noise = np.random.normal(0, 1, (n_samples, self.seq_len, self.n_features))
        
        conditions_batch = np.repeat(conditions.reshape(1, -1), n_samples, axis=0)
        
        h_fake = self.generator([noise, conditions_batch])
        
        x_fake = self.recovery(h_fake)
        
        return x_fake.numpy()


class FlameGAN:
    
    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.processor = FlameDataProcessor(data_dir)
        self.timegan = None
        self.norm_params = None
        self.seq_len = None
        
    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        print("Loading data files...")
        sequences, parameters = self.processor.load_all_data()
        
        if len(sequences) == 0:
            raise ValueError("No valid data files found!")
        
        print(f"Loaded {len(sequences)} files")
        
        seq_lengths = [seq.shape[0] for seq in sequences]
        self.seq_len = min(seq_lengths)
        print(f"Using sequence length: {self.seq_len}")
        
        sequences = np.array([seq[:self.seq_len] for seq in sequences])
        
        sequences_norm, self.norm_params = self.processor.normalize_data(sequences)
        
        print(f"Data shape: {sequences_norm.shape}")
        print(f"Parameters shape: {parameters.shape}")
        
        return sequences_norm, parameters
    
    def build_model(self, sequences: np.ndarray, parameters: np.ndarray):
        n_features = sequences.shape[2]
        n_conditions = parameters.shape[1]
        
        self.timegan = TimeGAN(
            seq_len=self.seq_len,
            n_features=n_features,
            n_conditions=n_conditions,
            hidden_dim=32,
            n_layers=3
        )
        
        print(f"Built TimeGAN model:")
        print(f"  Sequence length: {self.seq_len}")
        print(f"  Features: {n_features}")
        print(f"  Conditions: {n_conditions}")
    
    def train(self, sequences: np.ndarray, parameters: np.ndarray, 
              epochs: int = 1000, batch_size: int = 32):
        print(f"Training TimeGAN for {epochs} epochs...")
        n_batches = len(sequences) // batch_size + (1 if len(sequences) % batch_size != 0 else 0)
        
        for epoch in range(epochs):
            epoch_losses = {'e_loss': [], 'g_loss': [], 'd_loss': [], 'd_real': [], 'd_fake': []}
            indices = np.random.permutation(len(sequences))
            
            for i in range(0, len(sequences), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_data = sequences[batch_indices]
                batch_conditions = parameters[batch_indices]
                
                losses = self.timegan.train_step(
                    tf.convert_to_tensor(batch_data, dtype=tf.float32),
                    tf.convert_to_tensor(batch_conditions, dtype=tf.float32)
                )
                
                for key, value in losses.items():
                    epoch_losses[key].append(value.numpy())
            
            if epoch % 10 == 0:
                avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
                print(f"Epoch {epoch}/{epochs}: E_loss={avg_losses['e_loss']:.4f}, "
                      f"G_loss={avg_losses['g_loss']:.4f}, D_loss={avg_losses['d_loss']:.4f}, "
                      f"D_real={avg_losses['d_real']:.4f}, D_fake={avg_losses['d_fake']:.4f}")
    
    def generate_flame_data(self, phi: float, u: float, n_samples: int = 1) -> np.ndarray:
        if self.timegan is None:
            raise ValueError("Model not trained yet!")
        
        conditions = np.array([phi, u])
        synthetic_data = self.timegan.generate_samples(conditions, n_samples)
        
        synthetic_data = self.processor.denormalize_data(synthetic_data, self.norm_params)
        
        return synthetic_data
    
    def save_generated_data(self, data: np.ndarray, phi: float, u: float, 
                           duration: str = "generated", output_dir: str = None):
        if output_dir is None:
            output_dir = self.data_dir
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        phi_str = str(phi).replace('.', 'p')
        u_str = str(u).replace('.', 'p')
        
        filename = f"Phi_{phi_str}_u_{u_str}_{duration}_{timestamp}_generated.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            for row in data[0]:
                f.write(f"{row[0]:.6E}\t{row[1]:.6E}\t{row[2]:.6E}\n")
        
        print(f"Saved generated data to: {filename}")
        return filepath
    
    def visualize_comparison(self, real_data: np.ndarray, generated_data: np.ndarray, 
                           phi: float, u: float):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        feature_names = ['Heat', 'Time', 'Pressure']
        
        for i, (ax, feature) in enumerate(zip(axes, feature_names)):
            ax.plot(real_data[0, :, i], label='Real', alpha=0.7)
            ax.plot(generated_data[0, :, i], label='Generated', alpha=0.7)
            ax.set_title(f'{feature} (φ={phi}, u={u})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'comparison_phi_{phi}_u_{u}.png', dpi=150, bbox_inches='tight')
        plt.show()


def main():
    flame_gan = FlameGAN("./Baseline - Data")
    
    sequences, parameters = flame_gan.load_and_preprocess_data()
    
    flame_gan.build_model(sequences, parameters)
    
    flame_gan.train(sequences, parameters, epochs=500, batch_size=32)
    
    print("\nGenerating new flame prediction data...")
    
    test_params = [
        (0.8, 0.2),
        (0.8, 0.3),
        (0.8, 0.4),
    ]
    
    for phi, u in test_params:
        print(f"\nGenerating data for φ={phi}, u={u}")
        generated_data = flame_gan.generate_flame_data(phi, u, n_samples=1)
        
        flame_gan.save_generated_data(generated_data, phi, u, "10s")
        
        param_diffs = np.sum((parameters - np.array([phi, u]))**2, axis=1)
        closest_idx = np.argmin(param_diffs)
        closest_real = sequences[closest_idx:closest_idx+1]
        
        closest_real_denorm = flame_gan.processor.denormalize_data(
            closest_real, flame_gan.norm_params)
        
        flame_gan.visualize_comparison(closest_real_denorm, generated_data, phi, u)


if __name__ == "__main__":
    main()
