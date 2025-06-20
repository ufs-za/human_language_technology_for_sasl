# SASL HLT Artificial Intelligence Models

This folder contains AI models, training outputs, checkpoints (e.g., `.pkl` files), and reinforcement learning scripts developed for the South African Sign Language Human Language Technology (SASL HLT) project.

## Contents

- **Model Weights and Checkpoints**  
  Trained models saved as `.pkl` or `.pt` files for reuse and transfer learning.

- **Synthetic Data Generators**  
  Tools for creating artificial sign language data with variation in handshape, motion, and demographics.

- **Reinforcement Learning Scripts**  
  Experiments using RL frameworks to simulate learning environments for SASL recognition, translation, or avatar interaction.

## Usage Notes

- **Do not use for commercial purposes** without a signed data sharing and ethical use agreement.
- Ensure **proper academic attribution** when using models or code in publications.
- Refer to the root `LICENSE.md` for full usage terms (CC BY-NC-SA 4.0).

## Getting Started

1. Install required packages:
   ```bash
   pip install -r requirements.txt
````

2. Load a model:

   ```python
   import pickle

   with open('model_checkpoint.pkl', 'rb') as f:
       model = pickle.load(f)
   ```

3. Run a test script or connect to your preprocessing pipeline.

For advanced model integration, see the `notebooks/` folder for examples or `streamlit/` for web-based demos.

---

Maintained by the University of the Free State | ICDF HLT Collective
Contact: Interdisciplinary Centre for Digital Futures
