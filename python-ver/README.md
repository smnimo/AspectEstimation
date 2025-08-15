# AspectEstimation - Python Version

Complete Python implementation of the MATLAB AspectEstimation system with **4-layer pLSA**, **15-dimensional feature space**, **data processing pipeline**, and **visualization suite**.

## Overview

This is a comprehensive port of the MATLAB AspectEstimation codebase, featuring:

- **4-layer pLSA**: Session → Topic → Feature → Observation with fixed Feature→Observation mapping
- **15D Feature Space**: One-hot encoding of 5 attributes × 3 values each (corrected implementation)
- **Data Processing**: Complete LoadGaze.m and ReadwithTimestamp.m translation
- **Visualization**: Full plot_*.m series conversion with matplotlib/seaborn including additional gaze trajectories and correlation circles
- **Real Data Support**: Processing of actual experimental eye-tracking data
- **Exact Algorithm Equivalence**: Numerical results match MATLAB implementation

## 4-Layer Architecture

```
Session(s) → Topic(z) → Feature(v) → Observation(o)
     ↓           ↓           ↓           ↓
  P(z|s)     P(v|z)     P(o|v)     [Data]
  [Learn]    [Learn]    [Fixed]    [Input]
```

### Key Components

1. **Session → Topic**: `P(z|s)` - Learned session-topic distributions
2. **Topic → Feature**: `P(v|z)` - Learned topic-feature distributions (now 15-dimensional)
3. **Feature → Observation**: `P(o|v)` - **Fixed external mapping** (not learned)

**Critical**: 
- The Feature → Observation layer is externally provided from the knowledge base and remains fixed during training
- Features are now correctly represented as 15-dimensional one-hot vectors (5 attributes × 3 values each)
- Only the Session → Topic and Topic → Feature distributions are learned through the EM algorithm

## Features

### 1. Data Processing (`src/data/`)
- **Complete MATLAB Translation**: Exact LoadGaze.m + ReadwithTimestamp.m implementation
- **Eye Tracking**: Process gaze coordinate data (.csv files)
- **Layout Processing**: Handle item arrangements (.list files)  
- **Knowledge Integration**: Map items to semantic attributes (knowledge.csv)
- **Grid Mapping**: Convert coordinates to 3×4 item grid
- **Duration Filtering**: Apply 150ms minimum fixation threshold
- **Task Extraction**: Decode task numbers from layout encoding

### 2. 4-Layer pLSA Model (`src/models/`)
- **Probabilistic Architecture**: Session → Topic → Feature → Observation
- **EM Algorithm**: Iterative parameter estimation with exact MATLAB equivalence
- **Fixed Observation Layer**: External Feature→Observation mapping from knowledge base
- **Multiple Sessions**: Support for batch training across multiple sessions
- **Convergence Monitoring**: Likelihood tracking and early stopping

### 3. Visualization (`src/visualization/`)
- **Complete plot_*.m Translation**: All MATLAB plotting functions converted
- **Likelihood Analysis**: Model comparison across topic numbers (plot_L_topic.m)
- **Likelihood Convergence**: pLSA training progress visualization with EM iteration tracking
- **Confusion Matrices**: Task classification performance (plot_conf_matrix.m)
- **Duration Analysis**: User behavior patterns with boxplots (plot_dur_usr.m)
- **Aspect Matrices**: Feature-topic relationship visualization
- **Similarity Analysis**: Task and aspect correlation plots (plot_sim*.m)
- **Evaluation Metrics**: Precision, recall, F1-score visualization (plot_eval_AoF.m)
- **Gaze Trajectories**: Original eye movement path visualization (plot_org_gaze.m)
- **Aspect Correlations**: Circular correlation matrix plots (plot_aspect.R)
- **Comprehensive Reports**: Automated multi-plot generation with convergence tracking

### 4. Utilities (`src/utils/`)
- **Mathematical Functions**: Normalization, probability calculations
- **Data Preprocessing**: Array manipulation and validation
- **File I/O**: Robust data loading and saving utilities

## Installation

### 1. Clone Repository
```bash
git clone [repository_url]
cd AspectEstimation/python-ver
```

### 2. Virtual Environment
```bash
python -m venv aspectmodel
source aspectmodel/bin/activate  # Linux/WSL
# aspectmodel\Scripts\activate   # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage
```python
from src.data.gaze_reader import GazeDataReader
from src.models.plsa import FourLayerPLSA
from src.visualization.plotter import AspectPlotter

# 1. Load experimental data
reader = GazeDataReader(knowledge_csv_path='sample_data/knowledge.csv')
session = reader.load_session_from_files(
    gaze_csv_path='sample_data/170116024150092.csv',
    list_file_path='sample_data/170116024150092.list'
)

# 2. Train 4-layer pLSA model (15D features)
model = FourLayerPLSA(n_topics=3)
model.set_observation_model(reader.knowledge_data.values)
params, history = model.fit([session['semantic_attribute_seq_onehot']])  # 15D one-hot features

# 3. Generate visualizations
plotter = AspectPlotter()
plotter.create_comprehensive_report({
    'session': {(1, 1): session},
    'params': params,
    'likelihood_histories': {3: history}  # NEW: Include training convergence data
})
```

### Command Line Testing
```bash
# Test with real data files (15D feature processing)
python test_with_arguments.py \
  --knowledge sample_data/knowledge.csv \
  --gaze sample_data/170116024150092.csv \
  --layout sample_data/170116024150092.list

# Test comprehensive visualization pipeline (all 8 plot types)
python test_comprehensive_visualization.py

# Test complete 4-layer pLSA with 15D features
python test_four_layer.py

# Test actual data processing with 15D one-hot encoding
python test_actual_data.py
```

## Data Format

### Required Files

1. **knowledge.csv**: Feature→Observation mapping matrix
   ```
   Items × Attributes matrix (e.g., 12×5)
   Values: Integer attribute levels (1-3)
   Purpose: Defines fixed P(o|v) layer for 15D feature space
   Note: Each attribute value contributes to one-hot feature vector
   ```

2. **Gaze Data (.csv)**: Eye tracking recordings  
   ```
   Format: [timestamp, right_x, right_y, left_x, left_y]
   Coordinates: Normalized (0-1) screen coordinates
   Frequency: High-resolution temporal data
   ```

3. **Layout Data (.list)**: Item arrangement per session
   ```
   Format: Integer item IDs with task encoding
   Example: [101, 102, 103, 201, 202, ...]
   Encoding: 100s digit = task, 10s+1s = item ID
   ```

### Data Processing Pipeline

1. **Grid Mapping**: Convert gaze (x,y) → 3×4 grid positions
2. **Duration Filtering**: Apply 150ms minimum fixation threshold  
3. **Task Extraction**: Decode task numbers from layout (100s digit)
4. **Item Mapping**: Map grid positions to actual items via layout
5. **Semantic Conversion**: Generate 5D attribute vectors via knowledge base
6. **One-Hot Encoding**: Convert 5D attributes to 15D one-hot feature vectors (5 attributes × 3 values)
7. **Session Assembly**: Create temporal sequences for pLSA training with 15D features

## Visualization Capabilities

### Complete MATLAB plot_*.m Equivalence

The visualization system provides full translation of all MATLAB plotting functions:

#### 1. Likelihood Analysis (`plot_L_topic.m`)
- Compare model performance across different topic numbers
- **Training convergence monitoring with EM iteration tracking**
- **Individual and multi-model convergence visualization**
- Model selection support

#### 2. Confusion Matrices (`plot_conf_matrix.m`)  
- Task classification accuracy visualization
- Session-level performance analysis
- Heat map representations

#### 3. Duration Analysis (`plot_dur_usr.m`)
- User behavior pattern analysis
- Fixation duration distributions
- Boxplot visualizations across tasks/users

#### 4. Aspect Matrices
- Feature-topic relationship visualization (15×topics matrices)
- Probability distribution heat maps
- Topic interpretation support with one-hot feature dimensions

#### 5. Similarity Analysis (`plot_sim*.m`)
- Task similarity matrices
- Aspect correlation analysis  
- Hierarchical clustering visualizations

#### 6. Evaluation Metrics (`plot_eval_AoF.m`)
- Precision, recall, F1-score plots
- Performance metrics across conditions
- Statistical significance testing

#### 7. Gaze Trajectories (`plot_org_gaze.m`)
- Original eye movement path visualization
- Fixation point mapping over background images
- Start/end point highlighting
- Temporal sequence analysis

#### 8. Aspect Correlations (`plot_aspect.R`)
- Circular correlation matrix visualization for 15×topics matrices
- Feature-aspect relationship mapping with one-hot encoding
- Custom color schemes (red-white-blue)
- Matrix subset analysis for interpretable dimensions

### Sample Visualization Code
```python
from src.visualization.plotter import AspectPlotter

plotter = AspectPlotter()

# Individual plot types
plotter.plot_likelihood_comparison(models, histories)
plotter.plot_likelihood_convergence(likelihood_history, topic_num=3)  # NEW: EM convergence
plotter.plot_multiple_likelihood_convergence(likelihood_histories)     # NEW: Multi-model comparison
plotter.plot_confusion_matrix(true_labels, predicted_labels)
plotter.plot_duration_by_user(session_data)
plotter.plot_aspect_matrices(params['vz'])  # 15×topics matrices
plotter.plot_similarity_analysis(session_data)
plotter.plot_evaluation_metrics(metrics_data)

# Additional plot types (converted from MATLAB/R)
plotter.plot_original_gaze_trajectories(gaze_data)
plotter.plot_aspect_correlation_circles(aspect_matrix)  # 15D correlation circles

# Comprehensive report generation (10+ plot types including convergence tracking)
data_with_histories = {
    'session': session_data,
    'params': model_params,
    'likelihood_histories': {2: history_2topics, 3: history_3topics}  # NEW: Convergence data
}
plotter.create_comprehensive_report(data_with_histories)
```

## Testing

### Test Suite Overview

#### 🧪 Comprehensive Testing Suite
```bash
# Full system test with realistic data flow and convergence tracking
python test_comprehensive_visualization.py
```
**Purpose**: Complete end-to-end testing with realistic data flow
- **Data Flow**: KB → Item Sequences → Semantic Attributes → One-hot (realistic)
- **Scale**: 200 samples × 20 sessions (comprehensive)
- **Features**: All 10+ visualization functions including convergence tracking
- **pLSA**: Actual model training with realistic data (not mock)
- **Integration**: Complete realistic data pipeline testing
- **Use Case**: Production validation, release testing

#### ⚡ Quick Development Testing
```bash
# Lightweight test for likelihood convergence plotting only
python test_likelihood_convergence.py
```
**Purpose**: Quick validation during development
- **Data**: 50 samples × 3 sessions (lightweight)  
- **Features**: Likelihood convergence plotting only
- **pLSA**: Fast model training for feature validation
- **Integration**: Single feature testing
- **Use Case**: Feature development, debugging

#### 📋 Additional Test Files
```bash
# Test with real experimental data files
python test_with_arguments.py \
  --knowledge sample_data/knowledge.csv \
  --gaze sample_data/170116024150092.csv \
  --layout sample_data/170116024150092.list

# Core 4-layer pLSA functionality
python test_four_layer.py

# Real data processing pipeline
python test_actual_data.py
```

### Test File Roles

| Test File | Purpose | Data Scale | Speed | Use Case |
|-----------|---------|------------|-------|----------|
| `test_comprehensive_visualization.py` | **Complete system test with realistic data flow** | 200×20 sessions | Slow | Production validation |
| `test_likelihood_convergence.py` | **Quick feature test** | 50×3 sessions | Fast | Development validation |
| `test_with_arguments.py` | Real data testing | Variable | Medium | Data validation |
| `test_four_layer.py` | Core algorithm test | Small | Fast | Algorithm validation |
| `test_actual_data.py` | Data pipeline test | Variable | Medium | Pipeline validation |

### Recommended Testing Workflow

1. **During Development**: Use `test_likelihood_convergence.py` for quick validation
2. **Before Commits**: Run `test_comprehensive_visualization.py` for full validation  
3. **Real Data Testing**: Use `test_with_arguments.py` with actual data files
4. **Algorithm Changes**: Test with `test_four_layer.py` first

### Available Test Suite

```bash
# Core functionality with 15D features
python test_actual_data.py          # Real data processing pipeline with one-hot encoding
python test_four_layer.py           # 4-layer pLSA training validation (15D features)
python test_with_arguments.py       # Command-line interface testing

# Comprehensive visualization system  
python test_comprehensive_visualization.py  # Realistic data flow + all 10+ plotting functions including convergence

# Legacy individual tests (kept for compatibility)
python -m src.data.gaze_reader      # Data reader demonstration
python -m src.models.plsa           # Model training demonstration
```

### Sample Test Output

```
🚀 Starting Comprehensive Visualization Test Suite...
   📚 Using realistic data flow: KB → Items → Attributes → One-hot
   📈 Including pLSA likelihood convergence plotting

Testing realistic data generation with shared KB:
✅ Loaded knowledge base: (12, 5)
✅ Generated 20 sessions with realistic item sequences
✅ Created semantic attributes from KB lookup
✅ Generated 15D one-hot features: (20, 15, 200)

Training pLSA models with realistic data:
✅ Training 2-topic model: Final likelihood -47573.57 (8 iterations)
✅ Training 3-topic model: Final likelihood -44383.41 (8 iterations)
✅ Created likelihood convergence plots

Testing all visualization functions:
✅ Generated visualizations: 11 plots including convergence tracking
✅ All tests passed: Success Rate 100.0%
✅ Saved results: results/ and figures/
```

## Project Structure

```
python-ver/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── gaze_reader.py          # LoadGaze.m + ReadwithTimestamp.m (15D one-hot)
│   ├── models/
│   │   ├── __init__.py  
│   │   └── plsa.py                 # 4-layer pLSA implementation (15D compatible)
│   ├── utils/
│   │   ├── __init__.py
│   │   └── math_utils.py           # Mathematical utilities
│   └── visualization/
│       ├── __init__.py
│       └── plotter.py              # Complete plot_*.m + additional plots (15D aware)
├── figures/                        # Generated visualizations
├── results/                        # Model outputs (.npz files with 15D parameters)
├── sample_data/                    # Example experimental data
├── aspectmodel/                    # Virtual environment
├── requirements.txt                # Python dependencies
├── README.md                       # This documentation
├── test_actual_data.py             # Real data testing (15D features)
├── test_four_layer.py              # Core pLSA testing (15D features)
├── test_comprehensive_visualization.py  # All visualization functions
├── test_with_arguments.py          # Command-line interface
└── inspect_results.py              # Result analysis utility
```

## Key Achievements

### ✅ Complete MATLAB Translation
- **Exact Algorithm Equivalence**: All mathematical operations replicate MATLAB
- **15D Feature Space**: Corrected one-hot encoding implementation (5 attributes × 3 values)
- **Data Processing**: LoadGaze.m and ReadwithTimestamp.m fully translated with proper feature encoding
- **Model Architecture**: Correct 4-layer Session→Topic→Feature→Observation with 15D feature support
- **Visualization**: All plot_*.m functions converted with enhanced styling including gaze trajectories, correlation circles, and **training convergence tracking** (10+ plots total)

### ✅ Enhanced Functionality  
- **Realistic Data Flow**: KB → Item Sequences → Semantic Attributes → One-hot (production-ready)
- **Training Visualization**: Complete pLSA convergence monitoring with EM iteration tracking
- **Type Safety**: Full type hints and validation
- **Error Handling**: Robust error checking and recovery
- **Modular Design**: Clean separation of concerns
- **Comprehensive Testing**: Optimized test suite with realistic data flow validation
- **Modern Visualization**: matplotlib/seaborn with publication-quality output
- **Result Persistence**: All tests generate .npz result files for analysis

### ✅ Real Data Integration
- **Experimental Data**: Processes actual eye-tracking recordings with 15D feature encoding
- **Multiple Formats**: Handles .csv, .list, knowledge base files
- **Batch Processing**: Support for multiple sessions and participants
- **Statistical Analysis**: Complete evaluation metrics and comparisons
- **Result Storage**: Persistent .npz files for downstream analysis

## Dependencies

### Core Requirements
- **Python**: 3.8+ (tested with 3.12.3)
- **NumPy**: ≥ 1.21.0 (array operations)
- **SciPy**: ≥ 1.7.0 (statistical functions)
- **Pandas**: ≥ 1.3.0 (data manipulation)

### Visualization
- **Matplotlib**: ≥ 3.5.0 (plotting backend)
- **Seaborn**: ≥ 0.11.0 (statistical visualization)
- **Pillow**: ≥ 8.0.0 (image processing for gaze plots)

### Optional
- **Scikit-learn**: For additional machine learning utilities
- **Jupyter**: For interactive notebook development

## Differences from MATLAB Version

### Improvements
- **Enhanced Error Handling**: Comprehensive validation and recovery
- **Better Modularity**: Clear separation of data/models/visualization
- **Type Safety**: Full type annotations for reliability
- **Modern Visualization**: Publication-quality plots with matplotlib
- **Batch Processing**: Efficient handling of multiple sessions
- **Documentation**: Comprehensive inline documentation and examples

### Maintained Compatibility  
- **Numerical Equivalence**: Results match MATLAB to floating-point precision
- **Algorithm Fidelity**: Exact mathematical translation of EM algorithm
- **Data Format**: Compatible with existing MATLAB .mat files
- **Workflow**: Same experimental pipeline and analysis steps

## Performance Notes

### Optimization Features
- **Vectorized Operations**: NumPy-based efficient computation
- **Memory Management**: Optimized for large datasets
- **Early Stopping**: Automatic convergence detection
- **Batch Processing**: Efficient multi-session training

### Scaling Considerations
- **Sessions**: Tested with 10+ sessions simultaneously  
- **Topics**: Supports 2-10 topics efficiently
- **Features**: Handles 15-dimensional one-hot encoded feature space (extensible to larger dimensions)
- **Observations**: Processes sequences of 50-200 timepoints

## Troubleshooting

### Common Issues

1. **Data Loading Errors**:
   ```bash
   # Ensure correct file paths and formats
   python test_with_arguments.py --knowledge path/to/knowledge.csv ...
   ```

2. **Visualization Errors**:
   ```bash
   # Check matplotlib backend
   python -c "import matplotlib; print(matplotlib.get_backend())"
   ```

3. **Model Training Issues**:
   ```bash
   # Verify data preprocessing and 15D feature generation
   python test_actual_data.py
   ```

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality  
4. Ensure all tests pass
5. Submit pull request

### Code Style
- Follow PEP 8 conventions
- Add type hints for all functions
- Include comprehensive docstrings
- Maintain MATLAB algorithm equivalence

## License

[Specify license information]

## Citation

If you use this code in your research, please cite:

```bibtex
[Add appropriate citation format]
```

## Acknowledgments

This Python implementation is translated from the original MATLAB AspectEstimation codebase. Special thanks to the original authors for their comprehensive implementation of the 4-layer pLSA model and experimental data processing pipeline.

**Key Implementation Notes:**
- Corrected feature space implementation: 15-dimensional one-hot encoding (5 attributes × 3 values)
- Complete algorithm equivalence validated against MATLAB results
- Enhanced with modern Python practices while maintaining numerical precision
