from timegan_flame_generator import FlameGAN
import numpy as np
import matplotlib.pyplot as plt


def demo_complete_pipeline():
    print("=== TimeGAN Flame Prediction Data Generator Demo ===\n")
    
    print("Step 1: Initializing FlameGAN and loading data...")
    flame_gan = FlameGAN("./Baseline - Data")
    
    try:
        sequences, parameters = flame_gan.load_and_preprocess_data()
        print(f"✓ Successfully loaded {len(sequences)} data files")
        print(f"  Data shape: {sequences.shape}")
        print(f"  Parameter range - φ: {parameters[:, 0].min():.1f} to {parameters[:, 0].max():.1f}")
        print(f"  Parameter range - u: {parameters[:, 1].min():.1f} to {parameters[:, 1].max():.1f}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    print("\nStep 2: Building TimeGAN model...")
    flame_gan.build_model(sequences, parameters)
    print("✓ Model architecture built successfully")
    
    print("\nStep 3: Training model (demo with reduced epochs)...")
    flame_gan.train(sequences, parameters, epochs=200, batch_size=4)
    print("✓ Training completed")
    
    print("\nStep 4: Generating new flame prediction data...")
    
    test_cases = [
        {"phi": 1.2, "u": 0.5, "description": "Interpolated case 1"},
        {"phi": 1.2, "u": 0.6, "description": "Interpolated case 2"},
        {"phi": 1.2, "u": 0.7, "description": "Extrapolated case"}
    ]
    
    generated_files = []
    
    for i, case in enumerate(test_cases):
        phi, u = case["phi"], case["u"]
        print(f"\n  Generating case {i+1}: φ={phi}, u={u} ({case['description']})")
        
        try:
            generated_data = flame_gan.generate_flame_data(phi, u, n_samples=1)
            
            filepath = flame_gan.save_generated_data(generated_data, phi, u, "demo")
            generated_files.append(filepath)
            
            print(f"    ✓ Generated data shape: {generated_data.shape}")
            print(f"    ✓ Saved to: {filepath}")
            
        except Exception as e:
            print(f"    ✗ Error generating data: {e}")
    
    print("\nStep 5: Creating comparison visualizations...")
    
    for i, case in enumerate(test_cases):
        phi, u = case["phi"], case["u"]
        
        try:
            generated_data = flame_gan.generate_flame_data(phi, u, n_samples=1)
            
            param_diffs = np.sum((parameters - np.array([phi, u]))**2, axis=1)
            closest_idx = np.argmin(param_diffs)
            closest_real = sequences[closest_idx:closest_idx+1]
            
            closest_real_denorm = flame_gan.processor.denormalize_data(
                closest_real, flame_gan.norm_params)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            feature_names = ['Heat', 'Upstream Pressure', 'Downstream Pressure']
            colors_real = ['blue', 'green', 'red']
            colors_gen = ['lightblue', 'lightgreen', 'lightcoral']
            
            for j, (ax, feature, c_real, c_gen) in enumerate(zip(axes, feature_names, colors_real, colors_gen)):
                ax.plot(closest_real_denorm[0, :, j], label='Real (closest)', 
                       color=c_real, linewidth=2, alpha=0.8)
                ax.plot(generated_data[0, :, j], label='Generated', 
                       color=c_gen, linewidth=2, alpha=0.8)
                ax.set_title(f'{feature} (φ={phi}, u={u})')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'Case {i+1}: {case["description"]}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            plot_filename = f'demo_comparison_case_{i+1}_phi_{phi}_u_{u}.png'
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            print(f"    ✓ Saved comparison plot: {plot_filename}")
            plt.close()
            
        except Exception as e:
            print(f"    ✗ Error creating visualization: {e}")
    
    print("\n=== Demo Summary ===")
    print(f"✓ Trained TimeGAN model on {len(sequences)} flame prediction datasets")
    print(f"✓ Generated {len(generated_files)} new synthetic datasets")
    print("✓ Created comparison visualizations")
    print("\nGenerated files:")
    for filepath in generated_files:
        print(f"  - {filepath}")
    
    print(f"\nTo generate more data, use:")
    print(f"  from timegan_flame_generator import FlameGAN")
    print(f"  flame_gan = FlameGAN('.')")
    print(f"  # Load existing model and generate data")


def demo_parameter_space_exploration():
    print("\n=== Parameter Space Exploration Demo ===")
    
    flame_gan = FlameGAN("./Baseline - Data")
    sequences, parameters = flame_gan.load_and_preprocess_data()
    flame_gan.build_model(sequences, parameters)
    
    
    print("Training model for parameter space exploration...")
    flame_gan.train(sequences, parameters, epochs=150, batch_size=4)
    
    phi_range = np.linspace(0.7, 1.3, 5)
    u_range = np.linspace(0.2, 0.7, 5)
    
    print(f"Exploring parameter space:")
    print(f"  φ range: {phi_range}")
    print(f"  u range: {u_range}")
    
    results = []
    for phi in phi_range:
        for u in u_range:
            try:
                generated_data = flame_gan.generate_flame_data(phi, u, n_samples=1)
                results.append({
                    'phi': phi, 
                    'u': u, 
                    'data': generated_data[0],
                    'mean_heat': np.mean(generated_data[0, :, 0]),
                    'mean_pressure': np.mean(generated_data[0, :, 2])
                })
                print(f"  ✓ φ={phi:.1f}, u={u:.1f}: Heat={np.mean(generated_data[0, :, 0]):.2f}")
            except Exception as e:
                print(f"  ✗ φ={phi:.1f}, u={u:.1f}: Error - {e}")
    
    if results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        phi_vals = [r['phi'] for r in results]
        u_vals = [r['u'] for r in results]
        heat_vals = [r['mean_heat'] for r in results]
        pressure_vals = [r['mean_pressure'] for r in results]
        
        scatter1 = ax1.scatter(phi_vals, u_vals, c=heat_vals, cmap='viridis', s=100)
        ax1.set_xlabel('φ (Phi)')
        ax1.set_ylabel('u')
        ax1.set_title('Mean Heat across Parameter Space')
        plt.colorbar(scatter1, ax=ax1, label='Mean Heat')
        
        scatter2 = ax2.scatter(phi_vals, u_vals, c=pressure_vals, cmap='plasma', s=100)
        ax2.set_xlabel('φ (Phi)')
        ax2.set_ylabel('u')
        ax2.set_title('Mean Pressure across Parameter Space')
        plt.colorbar(scatter2, ax=ax2, label='Mean Pressure')
        
        plt.tight_layout()
        plt.savefig('parameter_space_exploration.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Parameter space visualization saved: parameter_space_exploration.png")
        plt.close()


if __name__ == "__main__":
    print("Starting TimeGAN Flame Prediction Demo...\n")
    
    demo_complete_pipeline()
    
    demo_parameter_space_exploration()
    
    print("\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("1. Use 'python generate_flame_data.py train' to train a production model")
    print("2. Use 'python generate_flame_data.py generate <phi> <u>' to generate specific data")
    print("3. Modify hyperparameters in timegan_flame_generator.py for better results")
