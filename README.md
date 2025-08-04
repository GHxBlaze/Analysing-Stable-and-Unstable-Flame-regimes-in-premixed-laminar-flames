# Flame Regime GAN Model

## Generating Oscillatory Flame Behavior with Deep Learning 

A Generative Adversarial Networks that creates oscillatory flame sequences based on combustion parameters(phi - equivalence ratio, u - velocity). This project implements a Generative Adversarial Network (GAN) to generate realistic oscillatory flame behavior sequences based on combustion parameters (φ - equivalence ratio, u - velocity parameter).

### Key Features:
- **Oscillatory Patterns**: Realistic flame oscillations with physics-based dynamics
- **Conditional Generation**: Parameter-controlled output(phi, u)
- **LSTM Architecture**: Temporal sequence modelling
- **Cloud Ready**: Optimized for Google Colab GPU/CPU

## Files Description

### Main Files
- `flame_gan_colab.py` - Main GAN implementation with all model classes
- `Flame_GAN_Colab.ipynb` - Complete Google Colab notebook with examples
- `test_flame_gan.py` - Local testing script
- `README_GAN.md` - This usage guide

### Model Architecture
- **Generator**: LSTM-based network that creates oscillatory sequences
- **Discriminator**: LSTM-based network that distinguishes real from fake sequences
- **Conditional Input**: φ and u parameters guide the generation process

## Usage Instructions

### 1. Google Colab Usage (Recommended)

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)

2. **Upload Files**: Upload `flame_gan_colab.py` to Colab

3. **Open Notebook**: Open `Flame_GAN_Colab.ipynb` in Colab

4. **Run Cells**: Execute cells in order:
   ```python
   # Install packages
   !pip install torch torchvision matplotlib seaborn pandas numpy scikit-learn tqdm
   
   # Import the model
   exec(open('flame_gan_colab.py').read())
   
   # Create and train GAN
   gan = FlameGAN()
   sequences, conditions = gan.create_oscillatory_data(n_samples=200)
   gan.train(sequences, conditions, epochs=150)
   ```

5. **Generate Sequences**:
   ```python
   # Generate for specific conditions
   stable_seq = gan.generate_sequence(phi=0.8, u=0.3)
   unstable_seq = gan.generate_sequence(phi=1.2, u=0.6)
   ```

### 2. Local Usage

1. **Install Dependencies**:
   ```bash
   pip install torch torchvision matplotlib seaborn pandas numpy scikit-learn tqdm scipy
   ```

2. **Run Test Script**:
   ```bash
   python test_flame_gan.py
   ```

### 3. Using Your Own Data

If you have real flame regime data:

```python
# Load your data
gan = FlameGAN()
sequences, conditions = gan.load_data_from_drive('/path/to/your/data')
gan.train(sequences, conditions, epochs=200)
```

## Model Parameters

### Flame Conditions
- **φ (phi)**: Equivalence ratio (0.8 - 1.2)
- **u**: Velocity parameter (0.2 - 0.7)

### Regime Classification
- **Stable**: φ < 0.9 OR (φ ≥ 1.0 AND u < 0.3)
- **Unstable**: Other combinations

### Network Architecture
- **Sequence Length**: 51 time steps
- **Input Dimensions**: 3 (X1, X2, X3)
- **Noise Dimension**: 100
- **Condition Dimension**: 2 (φ, u)

## Generated Sequences

The model generates oscillatory sequences with:
- **Realistic Frequencies**: Based on combustion physics
- **Proper Amplitudes**: Scaled according to regime
- **Phase Relationships**: Correlated between X1, X2, X3 components
- **Noise Characteristics**: Appropriate stochastic behavior

## Example Outputs

### Stable Regime (φ=0.8, u=0.3)
- Lower frequency oscillations
- Smaller amplitude variations
- More regular patterns

### Unstable Regime (φ=1.2, u=0.6)  
- Higher frequency oscillations
- Larger amplitude variations
- More chaotic patterns

## Visualization Features

The model includes several visualization tools:

1. **Time Series Plots**: Show oscillatory behavior over time
2. **Frequency Analysis**: FFT-based frequency domain analysis
3. **Phase Space Plots**: 2D and 3D phase space representations
4. **Parameter Studies**: Grid-based parameter sweep visualizations
5. **Training History**: Loss curves and convergence analysis

## Saving and Loading

```python
# Save trained model
gan.save_model('flame_gan_model.pth')

# Load model
gan = FlameGAN()
gan.load_model('flame_gan_model.pth')

# Generate new sequences
sequence = gan.generate_sequence(phi=1.0, u=0.5)
```

## File Formats

### Input Data Format
```
# Tab-separated values (TSV)
-6.334428E-2    -7.623805E-3    -6.469589E-1
-6.366636E-2    -1.567590E-2    -6.472810E-1
...
```

### Output Format
Generated sequences are saved in the same format as input data.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
   ```python
   gan = FlameGAN(device='cpu')
   ```

2. **Training Instability**: Adjust learning rates
   ```python
   # In FlameGAN.__init__()
   self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0001)
   self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0001)
   ```

3. **Poor Oscillations**: Increase training epochs or check data quality

### Performance Tips

- Use GPU for faster training: `device='cuda'`
- Start with smaller datasets for testing
- Monitor loss curves for convergence
- Experiment with different φ and u ranges

## Advanced Usage

### Custom Architecture
```python
# Modify network architecture
class CustomGenerator(nn.Module):
    def __init__(self, ...):
        # Your custom architecture
        pass

gan = FlameGAN()
gan.generator = CustomGenerator(...)
```

### Batch Generation
```python
# Generate multiple sequences
phi_values = [0.8, 0.9, 1.0, 1.1, 1.2]
u_values = [0.2, 0.3, 0.4, 0.5, 0.6]

for phi in phi_values:
    for u in u_values:
        seq = gan.generate_sequence(phi, u)
        filename = f"Phi_{phi:.1f}_u_{u:.1f}_generated.txt"
        np.savetxt(filename, seq, fmt='%.6E', delimiter='\t')
```

## Research Applications

This model can be used for:
- **Combustion Research**: Generate synthetic flame data
- **Control System Design**: Test flame control algorithms
- **Machine Learning**: Create training data for other models
- **Parameter Studies**: Explore flame behavior across parameter space

## Citation

If you use this model in your research, please cite:
```
Flame Regime GAN Model for Oscillatory Combustion Behavior Generation
[Your details here]
```

## Support

For issues and questions:
1. Check the Colab notebook examples
2. Run the test script locally
3. Review the troubleshooting section
4. Check parameter ranges and data formats

## Contact
- Created by Vaibhav Gangwar
Chemical Engineering | Shiv Nadar University
- Email: vg865@snu.edu.in




