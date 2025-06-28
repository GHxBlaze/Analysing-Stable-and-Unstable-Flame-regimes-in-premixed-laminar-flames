#  TimeGAN-Based Modeling of Flame Regimes in Laminar Premixed Combustion

This repository implements a **Time-Series Generative Adversarial Network (TimeGAN)** to analyze and generate synthetic data for **steady and unsteady flame regimes** in **laminar premixed combustion** systems. The goal is to model complex flame behavior—such as **FREI (Flames with Repetitive Extinction and Ignition)** and **Propagating Flames (PF)**—based on physical input conditions like equivalence ratio (ϕ) and flow velocity (u).

---

## Project Highlights

-  **Deep Learning for Flame Modeling**: Combines LSTM-based autoencoders and GANs for temporal generation.
-  **Conditioned Generation**: Model learns to produce flame dynamics based on input (ϕ, u).
-  **Physical Insight**: Enables visual and statistical comparison between real and synthetic flame behavior.
-  **Data-Driven Regime Analysis**: Useful for combustion diagnostics, regime prediction, and simulation speed-up.

---

##  Folder Structure
├── generate_flame_data.py # Script to train or generate data

├── demo_timegan.py # End-to-end demo pipeline with visualizations

├── timegan_flame_generator.py # Full TimeGAN model class and training loop

├── test_components.py # Unit tests for file parsing and loaders

├── /Baseline - Data/ # Experimental flame data (heat, pressure, time)

├── /Generated Data/ # Model-generated synthetic data

├── /Visualizations/ # PNG plots comparing real and generated sequences


---

## Dataset Description

Each `.txt` file contains time series of:
- `Heat release rate`
- `Time`
- `Pressure`

File names include flame parameters, e.g.:
Phi_1p2_u_0p6_10s_20250118_161858.txt

From this, the model extracts `ϕ = 1.2` and `u = 0.6`.

---

##  Model: TimeGAN Overview

TimeGAN is a hybrid of supervised and adversarial learning:
- **Embedder**: Encodes time-series to latent space
- **Recovery**: Decodes latent back to real
- **Generator**: Produces synthetic sequences
- **Discriminator**: Differentiates real from generated
- **Supervisor**: Helps generator maintain time coherence

The model is trained using real sequences conditioned on `(ϕ, u)` and is capable of generating new synthetic sequences for unseen combinations.

---

##  How to Run

### 1. Train the Model
```bash
python generate_flame_data.py train
```
### 2. Generate New Synthetic Flame Data
```bash
python generate_flame_data.py generate <phi> <u> [duration]
```
Example:
```bash
python generate_flame_data.py generate 1.0 0.4 10s
```
### 3. Run the Demo Pipeline
```bash
python demo_timegan.py
```
## Results
The demo script will generate comparative plots showing:
- Real vs. generated heat release

- Upstream/Downstream pressure dynamics

- Time evolution of unstable regimes (FREI, PF)

## Contact
- Created by Vaibhav Gangwar
Chemical Engineering | Shiv Nadar University
- Email: vg865@snu.edu.in
