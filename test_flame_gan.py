"""
Test script for Flame GAN Model
Run this to test the GAN with synthetic oscillatory data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flame_gan_colab import FlameGAN, demo_with_synthetic_data
import matplotlib.pyplot as plt
import numpy as np

def quick_test():
    """Quick test of the GAN model"""
    print("Testing Flame GAN Model")
    print("=" * 50)
    
    try:
        # Create GAN instance
        gan = FlameGAN(device='cpu')  # Use CPU for testing
        
        # Create a small amount of synthetic data for quick test
        print("Creating synthetic oscillatory data...")
        sequences, conditions = gan.create_oscillatory_data(n_samples=50)
        
        # Show sample data
        print(f"Generated {len(sequences)} sequences with shape {sequences.shape}")
        
        # Plot a few examples
        plt.figure(figsize=(15, 10))
        for i in range(6):
            plt.subplot(2, 3, i+1)
            seq = sequences[i]
            cond = conditions[i]
            
            # Plot all three components
            plt.plot(seq[:, 0], label='X1', alpha=0.8, linewidth=2)
            plt.plot(seq[:, 1], label='X2', alpha=0.8, linewidth=2) 
            plt.plot(seq[:, 2], label='X3', alpha=0.8, linewidth=2)
            
            regime = "Stable" if cond[0] < 0.9 or (cond[0] >= 1.0 and cond[1] < 0.3) else "Unstable"
            plt.title(f'{regime}: φ={cond[0]:.1f}, u={cond[1]:.1f}')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('Synthetic Oscillatory Flame Data', y=1.02, fontsize=16)
        plt.savefig('synthetic_flame_data.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Quick training test (just a few epochs)
        print("Quick training test...")
        gan.train(sequences, conditions, epochs=20, batch_size=8, save_interval=10)
        
        # Test generation
        print("Testing sequence generation...")
        test_conditions = [(0.8, 0.3), (0.8, 0.6)]
        
        plt.figure(figsize=(12, 8))
        for i, (phi, u) in enumerate(test_conditions):
            generated_seq = gan.generate_sequence(phi, u)
            
            plt.subplot(2, 2, i*2 + 1)
            plt.plot(generated_seq[:, 0], 'r-', label='Heat Release', alpha=0.8, linewidth=2)
            plt.plot(generated_seq[:, 1], 'g-', label='Upstream Pressure', alpha=0.8, linewidth=2)
            plt.plot(generated_seq[:, 2], 'b-', label='Downstream Pressure', alpha=0.8, linewidth=2)
            
            regime = "Stable" if phi < 0.9 else "Unstable"
            plt.title(f'Generated {regime}: φ={phi}, u={u}')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Phase space plot
            plt.subplot(2, 2, i*2 + 2)
            plt.scatter(generated_seq[:, 0], generated_seq[:, 2], alpha=0.6, s=20)
            plt.xlabel('X1')
            plt.ylabel('X3')
            plt.title(f'Phase Space: φ={phi}, u={u}')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('generated_flame_sequences.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save model
        gan.save_model('flame_gan_test.pth')
        
        print("Test completed successfully!")
        print("Files created:")
        print("  - synthetic_flame_data.png")
        print("  - generated_flame_sequences.png") 
        print("  - flame_gan_test.pth")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_oscillations():
    """Analyze the oscillatory nature of generated sequences"""
    print("\nAnalyzing Oscillatory Behavior")
    print("=" * 40)
    
    try:
        gan = FlameGAN(device='cpu')
        
        # Create oscillatory training data
        sequences, conditions = gan.create_oscillatory_data(n_samples=100)
        
        # Quick training
        gan.train(sequences, conditions, epochs=30, batch_size=8)
        
        # Generate sequences for analysis
        test_cases = [
            (0.8, 0.2, "Stable Low"),
            (0.8, 0.6, "Stable High"),
            (1.0, 0.2, "Unstable Low"),
            (1.2, 0.3, "Unstable High")
        ]
        
        plt.figure(figsize=(16, 12))
        
        for i, (phi, u, label) in enumerate(test_cases):
            seq = gan.generate_sequence(phi, u)
            
            # Time series plot
            plt.subplot(4, 3, i*3 + 1)
            plt.plot(seq[:, 0], 'r-', alpha=0.8, linewidth=2, label='Heat Release')
            plt.plot(seq[:, 1], 'g-', alpha=0.8, linewidth=2, label='Upstrem Pressure')
            plt.plot(seq[:, 2], 'b-', alpha=0.8, linewidth=2, label='Downstream Pressure')
            plt.title(f'{label} (φ={phi}, u={u})')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Frequency analysis
            from scipy.fft import fft, fftfreq
            
            plt.subplot(4, 3, i*3 + 2)
            fft_x1 = np.abs(fft(seq[:, 0]))
            freqs = fftfreq(len(seq[:, 0]))
            pos_mask = freqs > 0
            plt.semilogy(freqs[pos_mask][:20], fft_x1[pos_mask][:20], 'r-', linewidth=2)
            plt.title(f'X1 Frequency Spectrum')
            plt.xlabel('Frequency')
            plt.ylabel('Magnitude')
            plt.grid(True, alpha=0.3)
            
            # Phase space
            plt.subplot(4, 3, i*3 + 3)
            plt.scatter(seq[:, 0], seq[:, 2], alpha=0.6, s=15, c=range(len(seq)), cmap='viridis')
            plt.xlabel('X1')
            plt.ylabel('X3')
            plt.title('Phase Space (X1 vs X3)')
            plt.grid(True, alpha=0.3)
            plt.colorbar(label='Time')
        
        plt.tight_layout()
        plt.savefig('oscillation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Oscillation analysis completed!")
        print("Created: oscillation_analysis.png")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Flame GAN Test Suite")
    print("=" * 60)
    
    # Run quick test
    success = quick_test()
    
    if success:
        # Run oscillation analysis
        analyze_oscillations()
        
        print("\nAll tests completed successfully!")
        print("\nTo use in Google Colab:")
        print("1. Upload flame_gan_colab.py to Colab")
        print("2. Open Flame_GAN_Colab.ipynb")
        print("3. Run all cells")
        print("4. Upload your flame data to Google Drive if available")
    else:
        print("\nTests failed. Check error messages above.")
