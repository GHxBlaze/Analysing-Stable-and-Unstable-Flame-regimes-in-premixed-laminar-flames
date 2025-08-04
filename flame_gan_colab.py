"""
Flame Regime GAN Model for Google Colab
Generates oscillatory flame behavior using Generative Adversarial Networks
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import re
import glob
import os
from typing import Tuple, List, Dict, Optional
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class FlameDataset(Dataset):
    """Custom dataset for flame regime data"""
    
    def __init__(self, sequences, conditions, sequence_length=51):
        self.sequences = torch.FloatTensor(sequences)
        self.conditions = torch.FloatTensor(conditions)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.conditions[idx]

class Generator(nn.Module):
    """Generator network for creating oscillatory flame sequences"""
    
    def __init__(self, noise_dim=100, condition_dim=2, sequence_length=51, output_dim=3):
        super(Generator, self).__init__()
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        self.condition_dim = condition_dim
        
        # Combine noise and conditions
        self.input_dim = noise_dim + condition_dim
        
        # Initial fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # LSTM layers for temporal dynamics
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim),
            nn.Tanh()  # Output in range [-1, 1]
        )
        
    def forward(self, noise, conditions):
        batch_size = noise.size(0)
        
        # Concatenate noise and conditions
        x = torch.cat([noise, conditions], dim=1)
        
        # Pass through FC layers
        x = self.fc1(x)
        x = self.fc2(x)
        
        # Expand for sequence generation
        x = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Generate output sequence
        output = self.output_layer(lstm_out)
        
        return output

class Discriminator(nn.Module):
    """Discriminator network for distinguishing real from fake sequences"""
    
    def __init__(self, sequence_length=51, input_dim=3, condition_dim=2):
        super(Discriminator, self).__init__()
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        
        # LSTM for processing sequences
        self.lstm = nn.LSTM(
            input_size=input_dim + condition_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, sequences, conditions):
        batch_size = sequences.size(0)
        
        # Expand conditions to match sequence length
        conditions_expanded = conditions.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # Concatenate sequences with conditions
        x = torch.cat([sequences, conditions_expanded], dim=2)
        
        # Pass through LSTM
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Use the last hidden state for classification
        output = self.classifier(hidden[-1])
        
        return output

class FlameGAN:
    """Main GAN class for flame regime modeling"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.sequence_length = 51
        self.output_dim = 3  # X1, X2, X3 or Y1, Y2, Y3
        self.condition_dim = 2  # phi, u
        self.noise_dim = 100
        
        # Initialize networks
        self.generator = Generator(
            noise_dim=self.noise_dim,
            condition_dim=self.condition_dim,
            sequence_length=self.sequence_length,
            output_dim=self.output_dim
        ).to(self.device)
        
        self.discriminator = Discriminator(
            sequence_length=self.sequence_length,
            input_dim=self.output_dim,
            condition_dim=self.condition_dim
        ).to(self.device)
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Scalers for data normalization
        self.sequence_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.condition_scaler = StandardScaler()
        
        # Training history
        self.training_history = {
            'g_losses': [],
            'd_losses': [],
            'epochs': []
        }
        
    def parse_filename(self, filename: str) -> Dict[str, float]:
        """Extract parameters from filename"""
        pattern = r'Phi_(\d+)p(\d+)_u_(\d+)p(\d+)_(\d+)s'
        match = re.search(pattern, filename)
        
        if match:
            phi_int, phi_dec, u_int, u_dec, duration = match.groups()
            phi = float(f"{phi_int}.{phi_dec}")
            u = float(f"{u_int}.{u_dec}")
            duration = int(duration)
            
            return {
                'phi': phi,
                'u': u,
                'duration': duration
            }
        else:
            raise ValueError(f"Cannot parse filename: {filename}")
    
    def load_data_from_drive(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from Google Drive
        Expected structure: data_path contains folders with flame regime data
        """
        sequences = []
        conditions = []
        
        # Load stable regime data
        stable_input_path = os.path.join(data_path, "Stable Flame Regime", "Stable Flame Regime", "Input Dataset")
        stable_output_path = os.path.join(data_path, "Stable Flame Regime", "Stable Flame Regime", "Output Dataset")
        
        if os.path.exists(stable_input_path):
            input_files = glob.glob(os.path.join(stable_input_path, "*.txt"))
            
            for input_file in input_files:
                try:
                    params = self.parse_filename(os.path.basename(input_file))
                    
                    # Load input data
                    input_data = np.loadtxt(input_file)
                    
                    # Find corresponding output file
                    phi_str = f"{params['phi']:.1f}".replace('.', 'p')
                    u_str = f"{params['u']:.2f}".replace('.', 'p')
                    output_pattern = f"Phi_{phi_str}_u_{u_str}_demo_*_generated.txt"
                    output_files = glob.glob(os.path.join(stable_output_path, output_pattern))
                    
                    if output_files:
                        output_data = np.loadtxt(output_files[0])
                        
                        # Ensure consistent length
                        min_len = min(len(input_data), len(output_data), self.sequence_length)
                        
                        # Add input sequence
                        sequences.append(input_data[:min_len])
                        conditions.append([params['phi'], params['u']])
                        
                        # Add output sequence
                        sequences.append(output_data[:min_len])
                        conditions.append([params['phi'], params['u']])
                        
                        print(f"Loaded: {os.path.basename(input_file)}")
                        
                except Exception as e:
                    print(f"Error loading {input_file}: {e}")
        
        # Load unstable regime data
        unstable_input_path = os.path.join(data_path, "Unstable Flame Regime", "Unstable Flame Regime", "Input Data")
        unstable_output_path = os.path.join(data_path, "Unstable Flame Regime", "Unstable Flame Regime", "Output Data")
        
        if os.path.exists(unstable_input_path):
            input_files = glob.glob(os.path.join(unstable_input_path, "*.txt"))
            
            for input_file in input_files:
                try:
                    params = self.parse_filename(os.path.basename(input_file))
                    
                    # Load input data
                    input_data = np.loadtxt(input_file)
                    
                    # Find corresponding output file
                    phi_str = f"{params['phi']:.1f}".replace('.', 'p')
                    u_str = f"{params['u']:.2f}".replace('.', 'p')
                    output_pattern = f"Phi_{phi_str}_u_{u_str}_demo_*_generated.txt"
                    output_files = glob.glob(os.path.join(unstable_output_path, output_pattern))
                    
                    if output_files:
                        output_data = np.loadtxt(output_files[0])
                        
                        # Ensure consistent length
                        min_len = min(len(input_data), len(output_data), self.sequence_length)
                        
                        # Add input sequence
                        sequences.append(input_data[:min_len])
                        conditions.append([params['phi'], params['u']])
                        
                        # Add output sequence  
                        sequences.append(output_data[:min_len])
                        conditions.append([params['phi'], params['u']])
                        
                        print(f"Loaded: {os.path.basename(input_file)}")
                        
                except Exception as e:
                    print(f"Error loading {input_file}: {e}")
        
        if len(sequences) == 0:
            raise ValueError("No data could be loaded! Check your data path.")
        
        # Convert to numpy arrays and pad/truncate to consistent length
        max_len = self.sequence_length
        padded_sequences = []
        
        for seq in sequences:
            if len(seq) < max_len:
                # Pad with last value
                padded = np.pad(seq, ((0, max_len - len(seq)), (0, 0)), mode='edge')
            else:
                # Truncate
                padded = seq[:max_len]
            padded_sequences.append(padded)
        
        sequences = np.array(padded_sequences)
        conditions = np.array(conditions)
        
        print(f"Loaded {len(sequences)} sequences with shape {sequences.shape}")
        print(f"Conditions shape: {conditions.shape}")
        
        return sequences, conditions
    
    def create_oscillatory_data(self, n_samples=1000):
        """
        Create synthetic oscillatory data for training when real data is not available
        """
        sequences = []
        conditions = []
        
        # Parameter ranges
        phi_range = [0.8, 0.9, 1.0, 1.1, 1.2]
        u_range = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        
        for _ in range(n_samples):
            phi = np.random.choice(phi_range)
            u = np.random.choice(u_range)
            
            # Create oscillatory sequence
            t = np.linspace(0, 4*np.pi, self.sequence_length)
            
            # Base frequencies depend on regime
            if phi < 0.9:  # Stable regime
                base_freq = 1.0 + u * 2.0
                amplitude = 0.1 + phi * 0.05
            else:  # Unstable regime
                base_freq = 2.0 + u * 3.0
                amplitude = 0.2 + phi * 0.1
            
            # Generate oscillatory components
            x1 = amplitude * np.sin(base_freq * t + np.random.uniform(0, 2*np.pi))
            x1 += 0.05 * amplitude * np.sin(3 * base_freq * t + np.random.uniform(0, 2*np.pi))
            x1 += 0.02 * np.random.randn(self.sequence_length)
            
            x2 = 0.7 * amplitude * np.cos(1.3 * base_freq * t + np.random.uniform(0, 2*np.pi))
            x2 += 0.03 * amplitude * np.sin(2.7 * base_freq * t + np.random.uniform(0, 2*np.pi))
            x2 += 0.015 * np.random.randn(self.sequence_length)
            
            x3 = 1.2 * amplitude * np.sin(0.8 * base_freq * t + np.random.uniform(0, 2*np.pi))
            x3 += 0.4 * amplitude * np.cos(2.1 * base_freq * t + np.random.uniform(0, 2*np.pi))
            x3 += 0.025 * np.random.randn(self.sequence_length)
            
            # Add regime-specific offsets
            x1 += -0.06 if phi < 0.9 else -0.07
            x2 += -0.01 if phi < 0.9 else -0.012
            x3 += -0.64 if phi < 0.9 else 0.028
            
            sequence = np.column_stack([x1, x2, x3])
            sequences.append(sequence)
            conditions.append([phi, u])
        
        return np.array(sequences), np.array(conditions)
    
    def prepare_data(self, sequences, conditions, test_size=0.2):
        """Prepare and normalize data for training"""
        
        # Reshape sequences for scaling
        n_samples, seq_len, n_features = sequences.shape
        sequences_reshaped = sequences.reshape(-1, n_features)
        
        # Fit and transform sequences
        sequences_normalized = self.sequence_scaler.fit_transform(sequences_reshaped)
        sequences_normalized = sequences_normalized.reshape(n_samples, seq_len, n_features)
        
        # Normalize conditions
        conditions_normalized = self.condition_scaler.fit_transform(conditions)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            sequences_normalized, conditions_normalized,
            test_size=test_size, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def train(self, sequences, conditions, epochs=200, batch_size=32, save_interval=50):
        """Train the GAN"""
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(sequences, conditions)
        
        # Create dataset and dataloader
        train_dataset = FlameDataset(X_train, y_train, self.sequence_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Training on {len(X_train)} samples")
        print(f"Device: {self.device}")
        
        for epoch in range(epochs):
            g_losses = []
            d_losses = []
            
            for batch_idx, (real_sequences, real_conditions) in enumerate(train_loader):
                batch_size = real_sequences.size(0)
                
                # Move to device
                real_sequences = real_sequences.to(self.device)
                real_conditions = real_conditions.to(self.device)
                
                # Labels
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                
                self.d_optimizer.zero_grad()
                
                # Real sequences
                real_pred = self.discriminator(real_sequences, real_conditions)
                d_real_loss = self.criterion(real_pred, real_labels)
                
                # Fake sequences
                noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                fake_sequences = self.generator(noise, real_conditions)
                fake_pred = self.discriminator(fake_sequences.detach(), real_conditions)
                d_fake_loss = self.criterion(fake_pred, fake_labels)
                
                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                self.d_optimizer.step()
                
                # -----------------
                #  Train Generator
                # -----------------
                
                self.g_optimizer.zero_grad()
                
                # Generate fake sequences
                noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                fake_sequences = self.generator(noise, real_conditions)
                
                # Generator loss
                fake_pred = self.discriminator(fake_sequences, real_conditions)
                g_loss = self.criterion(fake_pred, real_labels)
                
                g_loss.backward()
                self.g_optimizer.step()
                
                # Store losses
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
            
            # Average losses for the epoch
            avg_g_loss = np.mean(g_losses)
            avg_d_loss = np.mean(d_losses)
            
            self.training_history['g_losses'].append(avg_g_loss)
            self.training_history['d_losses'].append(avg_d_loss)
            self.training_history['epochs'].append(epoch)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}] - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")
            
            # Save model periodically
            if epoch % save_interval == 0 and epoch > 0:
                self.save_model(f"flame_gan_epoch_{epoch}.pth")
        
        print("Training completed!")
    
    def generate_sequence(self, phi: float, u: float, n_samples: int = 1) -> np.ndarray:
        """Generate flame sequence for given conditions"""
        
        self.generator.eval()
        
        with torch.no_grad():
            # Prepare conditions
            conditions = np.array([[phi, u]] * n_samples)
            conditions_normalized = self.condition_scaler.transform(conditions)
            conditions_tensor = torch.FloatTensor(conditions_normalized).to(self.device)
            
            # Generate noise
            noise = torch.randn(n_samples, self.noise_dim).to(self.device)
            
            # Generate sequences
            fake_sequences = self.generator(noise, conditions_tensor)
            
            # Convert back to numpy and denormalize
            fake_sequences = fake_sequences.cpu().numpy()
            
            # Reshape for inverse transform
            n_samples, seq_len, n_features = fake_sequences.shape
            fake_sequences_reshaped = fake_sequences.reshape(-1, n_features)
            
            # Denormalize
            fake_sequences_denorm = self.sequence_scaler.inverse_transform(fake_sequences_reshaped)
            fake_sequences_denorm = fake_sequences_denorm.reshape(n_samples, seq_len, n_features)
            
        return fake_sequences_denorm[0] if n_samples == 1 else fake_sequences_denorm
    
    def plot_training_history(self):
        """Plot training losses"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['epochs'], self.training_history['g_losses'], label='Generator')
        plt.plot(self.training_history['epochs'], self.training_history['d_losses'], label='Discriminator')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.training_history['epochs'][-50:], self.training_history['g_losses'][-50:], label='Generator')
        plt.plot(self.training_history['epochs'][-50:], self.training_history['d_losses'][-50:], label='Discriminator')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses (Last 50 epochs)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_generated_sequences(self, phi_values, u_values, save_path=None):
        """Plot generated sequences for different conditions"""
        
        n_conditions = len(phi_values)
        fig, axes = plt.subplots(n_conditions, 3, figsize=(15, 4*n_conditions))
        
        if n_conditions == 1:
            axes = axes.reshape(1, -1)
        
        for i, (phi, u) in enumerate(zip(phi_values, u_values)):
            # Generate sequence
            sequence = self.generate_sequence(phi, u)
            
            # Plot each component
            for j, (component, label) in enumerate(zip(['X1', 'X2', 'X3'], sequence.T)):
                axes[i, j].plot(component, linewidth=2, alpha=0.8)
                axes[i, j].set_title(f'φ={phi}, u={u} - {label}')
                axes[i, j].set_xlabel('Time Step')
                axes[i, j].set_ylabel('Value')
                axes[i, j].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'sequence_scaler': self.sequence_scaler,
            'condition_scaler': self.condition_scaler,
            'training_history': self.training_history,
            'sequence_length': self.sequence_length,
            'output_dim': self.output_dim,
            'condition_dim': self.condition_dim,
            'noise_dim': self.noise_dim
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        self.sequence_scaler = checkpoint['sequence_scaler']
        self.condition_scaler = checkpoint['condition_scaler']
        self.training_history = checkpoint['training_history']
        
        print(f"Model loaded from {filepath}")

# Example usage functions
def demo_with_synthetic_data():
    """Demonstrate the GAN with synthetic oscillatory data"""
    
    print("Creating Flame GAN with synthetic oscillatory data...")
    gan = FlameGAN()
    
    # Create synthetic oscillatory data
    print("Generating synthetic oscillatory training data...")
    sequences, conditions = gan.create_oscillatory_data(n_samples=500)
    
    # Train the model
    print("Training GAN...")
    gan.train(sequences, conditions, epochs=100, batch_size=16)
    
    # Plot training history
    gan.plot_training_history()
    
    # Generate and plot sequences for different conditions
    phi_values = [1.0, 1.0, 1.2]
    u_values = [0.1, 0.2, 0.1]
    
    print("Generating sequences for different conditions...")
    gan.plot_generated_sequences(phi_values, u_values, 'generated_sequences.png')
    
    # Save the model
    gan.save_model('flame_gan_synthetic.pth')
    
    return gan

def demo_with_real_data(data_path):
    """Demonstrate the GAN with real flame data from Google Drive"""
    
    print(f"Creating Flame GAN with real data from {data_path}...")
    gan = FlameGAN()
    
    try:
        # Load real data
        print("Loading real flame data...")
        sequences, conditions = gan.load_data_from_drive(data_path)
        
        # Train the model
        print("Training GAN on real data...")
        gan.train(sequences, conditions, epochs=200, batch_size=8)
        
    except Exception as e:
        print(f"Could not load real data: {e}")
        print("Falling back to synthetic data...")
        sequences, conditions = gan.create_oscillatory_data(n_samples=500)
        gan.train(sequences, conditions, epochs=100, batch_size=16)
    
    # Plot training history
    gan.plot_training_history()
    
    # Generate and plot sequences
    phi_values = [1.0, 1.0, 1.0, 1.2]
    u_values = [0.1, 0.2, 0.5, 0.1]
    
    gan.plot_generated_sequences(phi_values, u_values, 'real_data_sequences.png')
    
    # Save the model
    gan.save_model('flame_gan_real_data.pth')
    
    return gan
