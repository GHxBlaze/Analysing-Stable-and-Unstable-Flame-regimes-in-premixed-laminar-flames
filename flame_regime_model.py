"""
Flame Regime Data Model
Predicts flame behavior from input conditions and generates complete time series.
"""

import numpy as np
import pandas as pd
import os
import re
import glob
from typing import Tuple, List, Dict, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')


class FlameRegimeModel:
    """
    A comprehensive model for flame regime prediction and time series generation.
    """
    
    def __init__(self, data_root: str = "d:/chem"):
        self.data_root = data_root
        self.stable_input_path = os.path.join(data_root, "Stable Flame Regime", "Stable Flame Regime", "Input Dataset")
        self.stable_output_path = os.path.join(data_root, "Stable Flame Regime", "Stable Flame Regime", "Output Dataset")
        self.unstable_input_path = os.path.join(data_root, "Unstable Flame Regime", "Unstable Flame Regime", "Input Data")
        self.unstable_output_path = os.path.join(data_root, "Unstable Flame Regime", "Unstable Flame Regime", "Output Data")
        
        # Model components
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.feature_columns = ['phi', 'u', 'x1', 'x2', 'x3', 'time_normalized']
        self.target_columns = ['y1', 'y2', 'y3']
        
        # Data storage
        self.training_data = None
        self.is_trained = False
        
    def parse_filename(self, filename: str) -> Dict[str, float]:
        """
        Extract parameters from filename.
        
        Args:
            filename: Input filename like 'Phi_0p8_u_0p2_30s_20250118_152820.txt'
            
        Returns:
            Dictionary with phi, u, duration parameters
        """
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
                'duration': duration,
                'regime': 'stable' if 'Stable' in filename else 'unstable'
            }
        else:
            raise ValueError(f"Cannot parse filename: {filename}")
    
    def load_data_file(self, filepath: str) -> np.ndarray:
        """
        Load data from a single file.
        
        Args:
            filepath: Path to the data file
            
        Returns:
            Numpy array with the data
        """
        try:
            data = np.loadtxt(filepath)
            return data
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def create_time_series_features(self, data: np.ndarray, params: Dict) -> pd.DataFrame:
        """
        Create features from time series data.
        
        Args:
            data: Input time series data (N x 3)
            params: Parameters extracted from filename
            
        Returns:
            DataFrame with features
        """
        n_points = len(data)
        time_normalized = np.linspace(0, 1, n_points)
        
        features = pd.DataFrame({
            'phi': [params['phi']] * n_points,
            'u': [params['u']] * n_points,
            'x1': data[:, 0],
            'x2': data[:, 1],
            'x3': data[:, 2],
            'time_normalized': time_normalized,
            'regime': [1 if params['regime'] == 'stable' else 0] * n_points
        })
        
        return features
    
    def load_all_data(self) -> pd.DataFrame:
        """
        Load all training data from both stable and unstable regimes.
        
        Returns:
            Combined DataFrame with all training data
        """
        all_data = []
        
        # Load stable regime data
        stable_input_files = glob.glob(os.path.join(self.stable_input_path, "*.txt"))
        stable_output_files = glob.glob(os.path.join(self.stable_output_path, "*.txt"))
        
        for input_file in stable_input_files:
            try:
                # Find corresponding output file
                base_name = os.path.basename(input_file)
                params = self.parse_filename(base_name)
                  # Look for corresponding output file
                phi_str = f"{params['phi']:.1f}".replace('.', 'p')
                u_str = f"{params['u']:.2f}".replace('.', 'p')
                output_pattern = f"Phi_{phi_str}_u_{u_str}_demo_*_generated.txt"
                output_files = glob.glob(os.path.join(self.stable_output_path, output_pattern))
                
                if output_files:
                    input_data = self.load_data_file(input_file)
                    output_data = self.load_data_file(output_files[0])
                    
                    if input_data is not None and output_data is not None:
                        # Sample input data to match output length
                        if len(input_data) > len(output_data):
                            indices = np.linspace(0, len(input_data)-1, len(output_data), dtype=int)
                            input_data = input_data[indices]
                        
                        features = self.create_time_series_features(input_data, params)
                        features['y1'] = output_data[:, 0]
                        features['y2'] = output_data[:, 1]
                        features['y3'] = output_data[:, 2]
                        
                        all_data.append(features)
                        print(f"Loaded stable: {base_name}")
                        
            except Exception as e:
                print(f"Error processing {input_file}: {e}")
        
        # Load unstable regime data
        unstable_input_files = glob.glob(os.path.join(self.unstable_input_path, "*.txt"))
        unstable_output_files = glob.glob(os.path.join(self.unstable_output_path, "*.txt"))
        
        for input_file in unstable_input_files:
            try:
                base_name = os.path.basename(input_file)
                params = self.parse_filename(base_name)
                params['regime'] = 'unstable'
                  # Look for corresponding output file
                phi_str = f"{params['phi']:.1f}".replace('.', 'p')
                u_str = f"{params['u']:.2f}".replace('.', 'p')
                output_pattern = f"Phi_{phi_str}_u_{u_str}_demo_*_generated.txt"
                output_files = glob.glob(os.path.join(self.unstable_output_path, output_pattern))
                
                if output_files:
                    input_data = self.load_data_file(input_file)
                    output_data = self.load_data_file(output_files[0])
                    
                    if input_data is not None and output_data is not None:
                        # Sample input data to match output length
                        if len(input_data) > len(output_data):
                            indices = np.linspace(0, len(input_data)-1, len(output_data), dtype=int)
                            input_data = input_data[indices]
                        
                        features = self.create_time_series_features(input_data, params)
                        features['y1'] = output_data[:, 0]
                        features['y2'] = output_data[:, 1]
                        features['y3'] = output_data[:, 2]
                        
                        all_data.append(features)
                        print(f"Loaded unstable: {base_name}")
                        
            except Exception as e:
                print(f"Error processing {input_file}: {e}")
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"Total data points loaded: {len(combined_data)}")
            return combined_data
        else:
            raise ValueError("No data could be loaded!")
    
    def train_model(self, test_size: float = 0.2, random_state: int = 42):
        """
        Train the flame regime prediction model.
        
        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        print("Loading training data...")
        self.training_data = self.load_all_data()
        
        # Prepare features and targets
        X = self.training_data[self.feature_columns]
        y = self.training_data[self.target_columns]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        # Scale targets
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_test_scaled = self.scaler_y.transform(y_test)
          # Train model
        print("Training model...")
        self.model = MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=random_state,
                verbose=1
            )
        )
        
        self.model.fit(X_train_scaled, y_train_scaled)
        
        # Evaluate model
        y_pred_scaled = self.model.predict(X_test_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"MSE: {mse:.6f}")
        print(f"R² Score: {r2:.4f}")
        
        self.is_trained = True
        
    def generate_input_from_params(self, phi: float, u: float, n_points: int = 1000) -> np.ndarray:
        """
        Generate synthetic input data based on phi and u parameters.
        This creates a realistic input time series based on the patterns observed in training data.
        
        Args:
            phi: Equivalence ratio
            u: Velocity parameter
            n_points: Number of points to generate
            
        Returns:
            Generated input data (n_points x 3)
        """
        # Determine regime based on parameters (simplified heuristic)
        if phi < 0.9 or (phi >= 1.0 and u < 0.3):
            regime = 'stable'
        else:
            regime = 'unstable'
        
        # Generate base patterns based on observed data statistics
        t = np.linspace(0, 1, n_points)
        
        # Base amplitudes and frequencies based on regime and parameters
        if regime == 'stable':
            amp1 = -0.06 + 0.01 * np.random.randn()
            amp2 = -0.01 + 0.005 * np.random.randn()
            amp3 = -0.64 + 0.02 * (phi - 0.8) + 0.01 * np.random.randn()
            
            freq1 = 2 + u * 5
            freq2 = 3 + u * 8
            freq3 = 1 + u * 2
        else:
            amp1 = -0.07 + 0.015 * np.random.randn()
            amp2 = -0.012 + 0.008 * np.random.randn()
            amp3 = 0.028 + 0.01 * (phi - 1.0) + 0.005 * np.random.randn()
            
            freq1 = 3 + u * 8
            freq2 = 4 + u * 12
            freq3 = 2 + u * 6
        
        # Generate synthetic time series with realistic noise and trends
        x1 = amp1 + 0.005 * np.sin(freq1 * t) + 0.002 * np.random.randn(n_points)
        x2 = amp2 + 0.003 * np.sin(freq2 * t + np.pi/4) + 0.001 * np.random.randn(n_points)
        x3 = amp3 + 0.01 * np.sin(freq3 * t + np.pi/2) + 0.005 * np.random.randn(n_points)
        
        # Add some correlation between variables
        x1 += 0.1 * x3
        x2 += 0.05 * x1
        
        # Apply smoothing to make it more realistic
        x1 = savgol_filter(x1, window_length=min(51, n_points//10 if n_points//10 % 2 == 1 else n_points//10 + 1), polyorder=3)
        x2 = savgol_filter(x2, window_length=min(51, n_points//10 if n_points//10 % 2 == 1 else n_points//10 + 1), polyorder=3)
        x3 = savgol_filter(x3, window_length=min(51, n_points//10 if n_points//10 % 2 == 1 else n_points//10 + 1), polyorder=3)
        
        return np.column_stack([x1, x2, x3])
    
    def predict_series(self, filename: str, n_points: int = 51) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate complete time series prediction from filename.
        
        Args:
            filename: Input filename (e.g., 'Phi_0p8_u_0p2_30s_test.txt')
            n_points: Number of points in output series
            
        Returns:
            Tuple of (input_series, predicted_output_series)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
        
        # Parse parameters from filename
        params = self.parse_filename(filename)
        
        # Generate synthetic input data
        input_data = self.generate_input_from_params(params['phi'], params['u'], n_points)
        
        # Create features for prediction
        time_normalized = np.linspace(0, 1, n_points)
        regime_flag = 1 if params['phi'] < 0.9 or (params['phi'] >= 1.0 and params['u'] < 0.3) else 0
        
        X_pred = pd.DataFrame({
            'phi': [params['phi']] * n_points,
            'u': [params['u']] * n_points,
            'x1': input_data[:, 0],
            'x2': input_data[:, 1],
            'x3': input_data[:, 2],
            'time_normalized': time_normalized
        })
        
        # Make prediction
        X_pred_scaled = self.scaler_X.transform(X_pred[self.feature_columns])
        y_pred_scaled = self.model.predict(X_pred_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        return input_data, y_pred
    
    def save_model(self, filepath: str = "flame_regime_model.joblib"):
        """
        Save the trained model and scalers.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving!")
        
        model_data = {
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = "flame_regime_model.joblib"):
        """
        Load a trained model and scalers.
        
        Args:
            filepath: Path to load the model from
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler_X = model_data['scaler_X']
        self.scaler_y = model_data['scaler_y']
        self.feature_columns = model_data['feature_columns']
        self.target_columns = model_data['target_columns']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
    
    def plot_prediction(self, filename: str, save_path: Optional[str] = None):
        """
        Plot the prediction results.
        
        Args:
            filename: Input filename for prediction
            save_path: Optional path to save the plot
        """
        input_data, output_data = self.predict_series(filename)
        params = self.parse_filename(filename)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot input data
        ax1.plot(input_data[:, 0], label='X1', alpha=0.8)
        ax1.plot(input_data[:, 1], label='X2', alpha=0.8)
        ax1.plot(input_data[:, 2], label='X3', alpha=0.8)
        ax1.set_title(f'Generated Input Data (φ={params["phi"]}, u={params["u"]})')
        ax1.set_xlabel('Time Point')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot output data
        ax2.plot(output_data[:, 0], label='Y1', alpha=0.8)
        ax2.plot(output_data[:, 1], label='Y2', alpha=0.8)
        ax2.plot(output_data[:, 2], label='Y3', alpha=0.8)
        ax2.set_title('Predicted Output Data')
        ax2.set_xlabel('Time Point')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot phase space (X1 vs X3)
        ax3.scatter(input_data[:, 0], input_data[:, 2], alpha=0.6, s=20)
        ax3.set_title('Input Phase Space (X1 vs X3)')
        ax3.set_xlabel('X1')
        ax3.set_ylabel('X3')
        ax3.grid(True, alpha=0.3)
        
        # Plot phase space (Y1 vs Y3)
        ax4.scatter(output_data[:, 0], output_data[:, 2], alpha=0.6, s=20)
        ax4.set_title('Output Phase Space (Y1 vs Y3)')
        ax4.set_xlabel('Y1')
        ax4.set_ylabel('Y3')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Example usage
    model = FlameRegimeModel()
    
    # Train the model
    print("Training flame regime model...")
    model.train_model()
    
    # Save the model
    model.save_model("flame_regime_model.joblib")
    
    # Test prediction with a new filename
    test_filename = "Phi_0p9_u_0p4_15s_test.txt"
    print(f"\nGenerating prediction for: {test_filename}")
    
    input_series, output_series = model.predict_series(test_filename)
    
    print(f"Generated input series shape: {input_series.shape}")
    print(f"Generated output series shape: {output_series.shape}")
    
    # Plot the results
    model.plot_prediction(test_filename, "prediction_example.png")
