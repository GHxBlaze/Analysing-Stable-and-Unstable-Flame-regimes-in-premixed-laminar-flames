from timegan_flame_generator import FlameGAN
import numpy as np
import sys


def train_model():
    print("Initializing FlameGAN...")
    flame_gan = FlameGAN("./Baseline - Data")
    
    print("Loading and preprocessing data...")
    sequences, parameters = flame_gan.load_and_preprocess_data()
    
    print("Building model...")
    flame_gan.build_model(sequences, parameters)
    
    print("Training model...")
    flame_gan.train(sequences, parameters, epochs=300, batch_size=4)
    
    flame_gan.timegan.embedder.save_weights("embedder_weights.h5")
    flame_gan.timegan.recovery.save_weights("recovery_weights.h5")
    flame_gan.timegan.generator.save_weights("generator_weights.h5")
    
    np.save("norm_params.npy", flame_gan.norm_params)
    np.save("seq_len.npy", flame_gan.seq_len)
    
    print("Model training completed and saved!")
    return flame_gan


def generate_data(phi, u, duration="10s"):
    try:
        print(f"Loading trained model for generating φ={phi}, u={u}...")
        
        flame_gan = FlameGAN("./Baseline - Data")
        
        sequences, parameters = flame_gan.load_and_preprocess_data()
        flame_gan.build_model(sequences, parameters)
        
        flame_gan.timegan.embedder.load_weights("embedder_weights.h5")
        flame_gan.timegan.recovery.load_weights("recovery_weights.h5") 
        flame_gan.timegan.generator.load_weights("generator_weights.h5")
        
        flame_gan.norm_params = np.load("norm_params.npy", allow_pickle=True).item()
        flame_gan.seq_len = int(np.load("seq_len.npy"))
        
        generated_data = flame_gan.generate_flame_data(phi, u, n_samples=1)
        
        filepath = flame_gan.save_generated_data(generated_data, phi, u, duration)
        
        print(f"Successfully generated and saved flame data to: {filepath}")
        
    except FileNotFoundError as e:
        print("Trained model not found. Please run train_model() first.")
        print(f"Missing file: {e.filename}")
    except Exception as e:
        print(f"Error generating data: {e}")


def main():
    if len(sys.argv) < 2:
        print("\nFlame Prediction Data Generator")
        print("===============================")
        print("Usage:")
        print("  python generate_flame_data.py train")
        print("  python generate_flame_data.py generate <phi> <u> [duration]")
        print("\nExamples:")
        print("  python generate_flame_data.py train")
        print("  python generate_flame_data.py generate 0.9 0.35")
        print("  python generate_flame_data.py generate 1.1 0.55 15s")
        return
    
    command = sys.argv[1].lower()
    
    if command == "train":
        train_model()
        
    elif command == "generate":
        if len(sys.argv) < 4:
            print("Error: Missing phi and u parameters")
            print("Usage: python generate_flame_data.py generate <phi> <u> [duration]")
            return
        
        try:
            phi = float(sys.argv[2])
            u = float(sys.argv[3])
            duration = sys.argv[4] if len(sys.argv) > 4 else "10s"
            
            generate_data(phi, u, duration)
            
        except ValueError:
            print("Error: phi and u must be numeric values")
            
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, generate")


if __name__ == "__main__":
    main()
