# AspectEstimation - Python Implementation

This is a Python port of the MATLAB AspectEstimation code, implementing probabilistic Latent Semantic Analysis (pLSA) for aspect estimation from gaze tracking data.

## Original MATLAB Implementation

The original MATLAB code implements:
- pLSA (probabilistic Latent Semantic Analysis) for topic modeling
- Likelihood calculation for attribute sequences
- EM algorithm for parameter estimation
- Data processing for gaze tracking experiments

## Python Implementation Structure

```
python-ver/
├── aspectmodel/          # Virtual environment
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── plsa.py      # pLSA implementation (translated from plsa.m, estep.m, mstep.m)
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py # Data loading utilities (translated from LoadGaze.m, calc_lkh.m)
│   └── utils/
│       ├── __init__.py
│       └── math_utils.py  # Mathematical utilities (translated from normalize.m, multi_choose_prob.m)
├── main.py              # Main execution script
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Setup

1. Activate the virtual environment:
```bash
source ./aspectmodel/bin/activate  # Linux/Mac
# or
./aspectmodel/Scripts/activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Full Pipeline
```bash
python main.py --mode full
```

### Likelihood Calculation Only
```bash
python main.py --mode likelihood
```

### pLSA Training Only
```bash
python main.py --mode plsa --topics 5
```

## Key Components

### 1. pLSA Model (`src/models/plsa.py`)
- **PLSA class**: Main model implementation
- **fit()**: EM algorithm training (translated from `plsa.m`)
- **_e_step()**: Expectation step (translated from `estep.m`)
- **_m_step()**: Maximization step (translated from `mstep.m`)
- **train_multiple_runs()**: Multiple initialization training (translated from `train.m`)

### 2. Data Processing (`src/data/data_loader.py`)
- **GazeDataLoader**: Load and process gaze tracking data
- **LikelihoodCalculator**: Calculate likelihood for attribute sequences (translated from `calc_lkh.m`)
- **load_gazedata()**: Load MATLAB .mat files (equivalent to `load gazedata`)

### 3. Mathematical Utilities (`src/utils/math_utils.py`)
- **normalize()**: Matrix normalization (translated from `normalize.m`)
- **exact_prob()**: Exact probability calculation (translated from `multi_choose_prob.m`)
- **generate_all_pairs()**: Generate combination patterns (translated from allpair generation)

## Data Format

The Python implementation expects the same data format as the MATLAB version:
- `gazedata.mat`: Main data file containing semantic attributes and sequences
- `semantic_attributes`: Cell array of attribute values per session
- `semantic_attribute_seq`: Sequence of attribute observations

## Algorithm Details

### pLSA (Probabilistic Latent Semantic Analysis)
The implementation follows the standard pLSA algorithm:

1. **Initialization**: Random initialization of P(z|s) and P(v|z)
2. **E-step**: Calculate posterior probabilities P(z|v,s)
3. **M-step**: Update parameters to maximize likelihood
4. **Convergence**: Iterate until relative change in log-likelihood < threshold

### Likelihood Calculation
For sliding window analysis:
1. Generate all possible combinations for window size
2. Calculate exact probabilities for each combination
3. Apply sliding window over attribute sequences
4. Compute likelihood at each time point

## Differences from MATLAB Version

- **Array indexing**: Python uses 0-based indexing vs MATLAB's 1-based
- **Matrix operations**: Uses NumPy broadcasting instead of MATLAB's repmat
- **Data structures**: Python lists and dictionaries instead of MATLAB cell arrays
- **File I/O**: Uses scipy.io for .mat file compatibility
- **Error handling**: More robust error handling and validation

## Testing

The implementation includes sample data generation for testing when the original .mat files are not available.

## Performance Notes

- Uses NumPy for efficient numerical operations
- Vectorized operations where possible to maintain performance
- Memory-efficient implementation for large datasets

## Dependencies

- **numpy**: Numerical computing
- **scipy**: Scientific computing and .mat file I/O
- **matplotlib**: Plotting (for future visualization features)
- **pandas**: Data manipulation
- **scikit-learn**: Additional machine learning utilities

## Future Enhancements

- Visualization functions for results analysis
- Parameter tuning utilities
- Cross-validation functionality
- Integration with modern deep learning frameworks
