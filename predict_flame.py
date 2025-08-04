"""
Simple interface for using the Flame Regime Model
Usage: python predict_flame.py filename.txt
"""

import sys
import os
import numpy as np
from flame_regime_model import FlameRegimeModel
import argparse


def save_series_to_file(data: np.ndarray, filename: str, prefix: str = ""):
    """
    Save time series data to a text file in the same format as training data.
    
    Args:
        data: Numpy array to save
        filename: Original filename to base the output name on
        prefix: Prefix for the output filename
    """
    # Create output filename
    base_name = os.path.splitext(filename)[0]
    output_filename = f"{prefix}{base_name}_predicted.txt"
    
    # Save in scientific notation format (same as input data)
    np.savetxt(output_filename, data, fmt='%.6E', delimiter='\t')
    print(f"Saved to: {output_filename}")
    
    return output_filename


def main():
    parser = argparse.ArgumentParser(description='Predict flame regime time series from filename')
    parser.add_argument('filename', help='Input filename (e.g., Phi_0p8_u_0p2_30s_test.txt)')
    parser.add_argument('--model-path', default='flame_regime_model.joblib', 
                       help='Path to the trained model file')
    parser.add_argument('--n-points', type=int, default=51,
                       help='Number of points in the output series')
    parser.add_argument('--plot', action='store_true',
                       help='Generate and display plots')
    parser.add_argument('--save-input', action='store_true',
                       help='Save generated input series to file')
    parser.add_argument('--save-output', action='store_true',
                       help='Save predicted output series to file')
    
    args = parser.parse_args()
    
    # Initialize model
    model = FlameRegimeModel()
    
    # Load or train model
    if os.path.exists(args.model_path):
        print(f"Loading pre-trained model from {args.model_path}")
        model.load_model(args.model_path)
    else:
        print("No pre-trained model found. Training new model...")
        model.train_model()
        model.save_model(args.model_path)
    
    # Make prediction
    print(f"\nGenerating prediction for: {args.filename}")
    try:
        input_series, output_series = model.predict_series(args.filename, args.n_points)
        
        print(f"Successfully generated:")
        print(f"  Input series shape: {input_series.shape}")
        print(f"  Output series shape: {output_series.shape}")
        
        # Parse parameters for display
        params = model.parse_filename(args.filename)
        print(f"  Parameters: φ={params['phi']}, u={params['u']}")
        
        # Save files if requested
        if args.save_input:
            save_series_to_file(input_series, args.filename, "input_")
        
        if args.save_output:
            save_series_to_file(output_series, args.filename, "output_")
        
        # Generate plot if requested
        if args.plot:
            plot_filename = f"prediction_{os.path.splitext(args.filename)[0]}.png"
            model.plot_prediction(args.filename, plot_filename)
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"Input series:")
        print(f"  X1: mean={input_series[:, 0].mean():.6f}, std={input_series[:, 0].std():.6f}")
        print(f"  X2: mean={input_series[:, 1].mean():.6f}, std={input_series[:, 1].std():.6f}")
        print(f"  X3: mean={input_series[:, 2].mean():.6f}, std={input_series[:, 2].std():.6f}")
        
        print(f"Output series:")
        print(f"  Y1: mean={output_series[:, 0].mean():.6f}, std={output_series[:, 0].std():.6f}")
        print(f"  Y2: mean={output_series[:, 1].mean():.6f}, std={output_series[:, 1].std():.6f}")
        print(f"  Y3: mean={output_series[:, 2].mean():.6f}, std={output_series[:, 2].std():.6f}")
        
    except Exception as e:
        print(f"Error generating prediction: {e}")
        sys.exit(1)


def batch_predict():
    """
    Function to predict multiple files at once.
    """
    # Example filenames to test
    test_filenames = [
        "Phi_0p8_u_0p3_15s_test.txt",
        "Phi_0p9_u_0p4_20s_test.txt", 
        "Phi_1p0_u_0p5_10s_test.txt",
        "Phi_1p2_u_0p6_25s_test.txt"
    ]
    
    model = FlameRegimeModel()
    
    # Load or train model
    model_path = "flame_regime_model.joblib"
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        model.load_model(model_path)
    else:
        print("Training new model...")
        model.train_model()
        model.save_model(model_path)
    
    print("\nBatch prediction for test cases:")
    print("=" * 50)
    
    for filename in test_filenames:
        try:
            print(f"\nProcessing: {filename}")
            input_series, output_series = model.predict_series(filename, n_points=51)
            
            # Save both input and output
            save_series_to_file(input_series, filename, "input_")
            save_series_to_file(output_series, filename, "output_")
            
            # Generate plot
            plot_filename = f"prediction_{os.path.splitext(filename)[0]}.png"
            model.plot_prediction(filename, plot_filename)
            
            params = model.parse_filename(filename)
            print(f"  Parameters: φ={params['phi']}, u={params['u']}")
            print(f"  Generated {len(output_series)} time points")
            
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python predict_flame.py filename.txt")
        print("   or: python predict_flame.py --batch")
        print("\nFor batch prediction of test cases, use:")
        print("python predict_flame.py --batch")
        sys.exit(1)
    
    if "--batch" in sys.argv:
        batch_predict()
    else:
        main()
