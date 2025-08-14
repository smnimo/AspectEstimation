# AspectEstimation - Python Implementation

This is a Python port of the MATLAB AspectEstimation code, implementing a **four-layer probabilistic model** for aspect estimation.

## Four-Layer Architecture

This implementation uses the correct four-layer probabilistic model:

```
Session → Topic → Feature → Observation
```

### Key Architecture Details

1. **Session → Topic**: `P(z|s)` - Learned distribution mapping sessions to latent topics
2. **Topic → Feature**: `P(v|z)` - Learned distribution mapping topics to features  
3. **Feature → Observation**: `P(o|v)` - **Fixed external mapping** (not learned)

**Important**: The Feature → Observation layer is externally provided and remains fixed during training. Only the Session → Topic and Topic → Feature distributions are learned through the EM algorithm.

This follows the user specification: *"４層構造ですが，Session-->Topic-->Feature-->Observation です．Feature --> Observation が外部からの入力によって与えられ，その確率は学習時に固定（学習もなし）です．"*

## Quick Start

```bash
# Activate virtual environment
source aspectmodel/bin/activate

# Run four-layer demonstration
python main_four_layer.py

# Run with custom parameters
python main_four_layer.py --sessions 10 --features 20 --max-topics 5 --save

# Run tests
python test_four_layer.py
```

## Project Structure

```
python-ver/
├── aspectmodel/              # Virtual environment
├── src/
│   ├── models/
│   │   ├── plsa.py          # Four-layer pLSA implementation
│   │   └── segments.py      # Segment processing utilities
│   ├── data/
│   │   └── data_loader.py   # Data loading utilities
│   └── utils/
│       └── math_utils.py    # Mathematical utilities
├── main_four_layer.py       # Main demonstration script
├── main_four_layer_old.py   # Previous implementation (archived)
├── test_four_layer.py       # Unit tests for 4-layer model
├── results_four_layer/      # Generated results
└── requirements.txt         # Python dependencies
```

## Architecture Evolution

### ✅ Current Implementation
- **Session → Topic → Feature → Observation**
- Feature → Observation is **externally fixed** and provided
- Only 2 distributions are learned: P(z|s) and P(v|z)
- Mathematically correct four-layer model

### Previous Approach (main_four_layer.py)
- Session → Segment → Topic → Vocabulary
- Segment-based approach
- All layers were learned

## Core Classes

### `FourLayerPLSA`
The main class implementing the four-layer model:

```python
from src.models.plsa import FourLayerPLSA

# Initialize model
model = FourLayerPLSA(n_topics=3)

# Set fixed observation mapping (externally provided)
model.set_observation_model(p_o_v_matrix)

# Train model (learns only P(z|s) and P(v|z))
params, history = model.fit(sessions)
```

### Key Methods
- `set_observation_model(p_o_v)`: Set the fixed Feature → Observation mapping
- `fit(sessions)`: Train the model using EM algorithm
- Returns learned parameters: `{'zs': P(z|s), 'vz': [P(v|z) for each topic]}`

## Training Results

The model generates the following outputs:

- `P(z|s)`: Session-topic distributions (sessions × topics)
- `P(v|z)`: Topic-feature distributions (list of feature distributions per topic)
- Training likelihood histories showing convergence
- Parameter files (`.npz` format)
- CSV exports of topic-feature distributions for analysis

## Example Usage

```python
import numpy as np
from src.models.plsa import FourLayerPLSA

# Create sample data
sessions = [np.random.multinomial(1, [1/15]*15, 30).T for _ in range(5)]

# Create observation mapping (identity for simplicity)
p_o_v = np.eye(15)

# Train model
model = FourLayerPLSA(n_topics=2)
model.set_observation_model(p_o_v)
params, history = model.fit(sessions)

print(f"Session-topic shape: {params['zs'].shape}")
print(f"Number of topic distributions: {len(params['vz'])}")
print(f"Final likelihood: {history[-1]:.4f}")
```

## Dependencies

- Python 3.12+
- NumPy 2.3+
- SciPy
- matplotlib  
- pandas
- scikit-learn

## Virtual Environment

The project uses a dedicated virtual environment in `aspectmodel/`:

```bash
# Activate environment (Linux/WSL)
source aspectmodel/bin/activate

# Activate environment (Windows)
aspectmodel\Scripts\activate

# Install additional packages if needed
pip install package_name
```

## Testing

Run the test suite to verify the implementation:

```bash
# Test the corrected four-layer implementation
python test_four_layer.py

# Expected output:
# - Training completes successfully
# - Likelihood values improve over iterations
# - Parameter shapes are correct
# - Multiple topic configurations work
```

## Mathematical Foundation

The four-layer model optimizes:

```
P(o_t | s) = Σ_z Σ_v P(z|s) * P(v|z) * P(o_t|v)
```

Where:
- `P(z|s)`: Session → Topic (learned)
- `P(v|z)`: Topic → Feature (learned)  
- `P(o_t|v)`: Feature → Observation (fixed, externally provided)

The EM algorithm updates only the learnable parameters while keeping `P(o_t|v)` fixed throughout training.

## Translation from MATLAB

This Python implementation is translated from the original MATLAB files:
- `plsa.m` → `src/models/plsa.py`
- `estep.m`, `mstep.m` → EM algorithm methods
- `learn_aspects_from_mask.m` → Main training pipeline
- Mathematical utilities from various `.m` files

The key insight was recognizing that the MATLAB implementation used a four-layer model with a fixed observation layer, not a fully learnable four-layer model.
