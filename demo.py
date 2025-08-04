"""
Demonstration script for the Flame Regime Model
This script shows how to use the model and generates example predictions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from flame_regime_model import FlameRegimeModel


def demonstrate_model():
    """
    Demonstrate the flame regime model with example predictions.
    """
    print("Flame Regime Model Demonstration")
    print("=" * 40)
    
    # Initialize the model
    model = FlameRegimeModel()
    
    # Check if a trained model exists
    model_path = "flame_regime_model.joblib"
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model.load_model(model_path)
    else:
        print("Training new model...")
        print("This may take several minutes...")
        model.train_model()
        model.save_model(model_path)
    
    print("\nModel loaded successfully!")
    
    # Define test cases covering different regimes
    test_cases = [
        # Stable regime cases
        {
            'filename': 'Phi_0p8_u_0p2_30s_demo.txt',
            'description': 'Stable regime: Low phi, low velocity'
        },
        {
            'filename': 'Phi_0p8_u_0p3_25s_demo.txt', 
            'description': 'Stable regime: Medium phi, medium velocity'
        },
        # Unstable regime cases
        {
            'filename': 'Phi_1p0_u_0p3_20s_demo.txt',
            'description': 'Unstable regime: High phi, medium velocity'
        },
        {
            'filename': 'Phi_1p2_u_0p5_15s_demo.txt',
            'description': 'Unstable regime: High phi, high velocity'
        }
    ]
    
    print(f"\nGenerating predictions for {len(test_cases)} test cases:")
    print("-" * 60)
    
    # Generate predictions for each test case
    for i, case in enumerate(test_cases):
        print(f"\n{i+1}. {case['description']}")
        print(f"   Filename: {case['filename']}")
        
        try:
            # Generate prediction
            input_series, output_series = model.predict_series(case['filename'], n_points=51)
            
            # Extract parameters
            params = model.parse_filename(case['filename'])
            
            print(f"   Parameters: φ={params['phi']}, u={params['u']}")
            print(f"   Input shape: {input_series.shape}")
            print(f"   Output shape: {output_series.shape}")
            
            # Save the series to files
            input_filename = f"demo_input_{i+1}.txt"
            output_filename = f"demo_output_{i+1}.txt"
            
            np.savetxt(input_filename, input_series, fmt='%.6E', delimiter='\t')
            np.savetxt(output_filename, output_series, fmt='%.6E', delimiter='\t')
            
            print(f"   Saved input to: {input_filename}")
            print(f"   Saved output to: {output_filename}")
            
            # Generate plot
            plot_filename = f"demo_prediction_{i+1}.png"
            model.plot_prediction(case['filename'], plot_filename)
            print(f"   Plot saved to: {plot_filename}")
            
            # Print some statistics
            print(f"   Output statistics:")
            print(f"     Y1: [{output_series[:, 0].min():.3f}, {output_series[:, 0].max():.3f}]")
            print(f"     Y2: [{output_series[:, 1].min():.3f}, {output_series[:, 1].max():.3f}]")
            print(f"     Y3: [{output_series[:, 2].min():.3f}, {output_series[:, 2].max():.3f}]")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    print(f"\nDemonstration complete!")
    print(f"Generated files:")
    print(f"  - Model: {model_path}")
    print(f"  - Input series: demo_input_*.txt")
    print(f"  - Output series: demo_output_*.txt") 
    print(f"  - Plots: demo_prediction_*.png")


def compare_regimes():
    """
    Compare stable vs unstable regimes with similar parameters.
    """
    print("\n" + "=" * 50)
    print("REGIME COMPARISON")
    print("=" * 50)
    
    model = FlameRegimeModel()
    
    # Load the trained model
    if os.path.exists("flame_regime_model.joblib"):
        model.load_model("flame_regime_model.joblib")
    else:
        print("No trained model found. Please run the demonstration first.")
        return
    
    # Compare similar conditions in different regimes
    stable_case = 'Phi_0p8_u_0p4_20s_stable.txt'
    unstable_case = 'Phi_1p2_u_0p4_20s_unstable.txt'
    
    print(f"\nComparing:")
    print(f"  Stable: {stable_case}")
    print(f"  Unstable: {unstable_case}")
    
    try:
        # Generate predictions
        stable_input, stable_output = model.predict_series(stable_case, n_points=51)
        unstable_input, unstable_output = model.predict_series(unstable_case, n_points=51)
        
        # Create comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot input comparison
        ax1.plot(stable_input[:, 0], 'b-', label='Stable X1', alpha=0.7)
        ax1.plot(unstable_input[:, 0], 'r-', label='Unstable X1', alpha=0.7)
        ax1.set_title('Input Comparison (X1)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot output comparison
        ax2.plot(stable_output[:, 0], 'b-', label='Stable Y1', alpha=0.7)
        ax2.plot(unstable_output[:, 0], 'r-', label='Unstable Y1', alpha=0.7)
        ax2.set_title('Output Comparison (Y1)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot phase spaces
        ax3.scatter(stable_input[:, 0], stable_input[:, 2], c='blue', alpha=0.6, s=20, label='Stable')
        ax3.scatter(unstable_input[:, 0], unstable_input[:, 2], c='red', alpha=0.6, s=20, label='Unstable')
        ax3.set_title('Input Phase Space (X1 vs X3)')
        ax3.set_xlabel('X1')
        ax3.set_ylabel('X3')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4.scatter(stable_output[:, 0], stable_output[:, 2], c='blue', alpha=0.6, s=20, label='Stable')
        ax4.scatter(unstable_output[:, 0], unstable_output[:, 2], c='red', alpha=0.6, s=20, label='Unstable')
        ax4.set_title('Output Phase Space (Y1 vs Y3)')
        ax4.set_xlabel('Y1')
        ax4.set_ylabel('Y3')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('regime_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Comparison plot saved to: regime_comparison.png")
        
        # Print quantitative comparison
        print(f"\nQuantitative Comparison:")
        print(f"Stable regime (φ=0.8, u=0.4):")
        print(f"  Output range Y1: [{stable_output[:, 0].min():.3f}, {stable_output[:, 0].max():.3f}]")
        print(f"  Output std Y1: {stable_output[:, 0].std():.3f}")
        
        print(f"Unstable regime (φ=1.2, u=0.4):")
        print(f"  Output range Y1: [{unstable_output[:, 0].min():.3f}, {unstable_output[:, 0].max():.3f}]")
        print(f"  Output std Y1: {unstable_output[:, 0].std():.3f}")
        
    except Exception as e:
        print(f"Error in comparison: {e}")


if __name__ == "__main__":
    # Run the demonstration
    demonstrate_model()
    
    # Compare regimes
    compare_regimes()
    
    print(f"\n" + "=" * 60)
    print("USAGE INSTRUCTIONS")
    print("=" * 60)
    print("To use the model for new predictions:")
    print("1. Command line: python predict_flame.py Phi_0p8_u_0p3_20s_test.txt")
    print("2. In Python:")
    print("   from flame_regime_model import FlameRegimeModel")
    print("   model = FlameRegimeModel()")
    print("   model.load_model('flame_regime_model.joblib')")
    print("   input_data, output_data = model.predict_series('your_filename.txt')")
    print("3. Batch processing: python predict_flame.py --batch")
    print("\nFilename format: Phi_[phi]p[decimal]_u_[u]p[decimal]_[duration]s_[description].txt")
    print("Example: Phi_1p2_u_0p5_30s_experiment.txt → φ=1.2, u=0.5, 30s duration")
