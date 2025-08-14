# AspectEstimation - Python Version

Complete Python implementation of the MATLAB AspectEstimation system with **4-layer pLSA**, **data processing pipeline**, and **visualization suite**.

## Overview

This is a comprehensive port of the MATLAB AspectEstimation codebase, featuring:

- **4-layer pLSA**: Session → Topic → Feature → Observation with fixed Feature→Observation mapping
- **Data Processing**: Complete LoadGaze.m and ReadwithTimestamp.m translation
- **Visualization**: Full plot_*.m series conversion with matplotlib/seaborn
- **Real Data Support**: Processing of actual experimental eye-tracking data

## 4-Layer Architecture

```
Session(s) → Topic(z) → Feature(v) → Observation(o)
     ↓           ↓           ↓           ↓
  P(z|s)     P(v|z)     P(o|v)     [Data]
  [Learn]    [Learn]    [Fixed]    [Input]
```

### Key Components

1. **Session → Topic**: `P(z|s)` - Learned session-topic distributions
2. **Topic → Feature**: `P(v|z)` - Learned topic-feature distributions  
3. **Feature → Observation**: `P(o|v)` - **Fixed external mapping** (not learned)

**Critical**: The Feature → Observation layer is externally provided from the knowledge base and remains fixed during training. Only the Session → Topic and Topic → Feature distributions are learned through the EM algorithm.

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
- **Confusion Matrices**: Task classification performance (plot_conf_matrix.m)
- **Duration Analysis**: User behavior patterns with boxplots (plot_dur_usr.m)
- **Aspect Matrices**: Feature-topic relationship visualization
- **Similarity Analysis**: Task and aspect correlation plots (plot_sim*.m)
- **Evaluation Metrics**: Precision, recall, F1-score visualization (plot_eval_AoF.m)
- **Gaze Trajectories**: Original eye movement path visualization (plot_org_gaze.m)
- **Aspect Correlations**: Circular correlation matrix plots (plot_aspect.R)
- **Comprehensive Reports**: Automated multi-plot generation

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

# 2. Train 4-layer pLSA model
model = FourLayerPLSA(n_topics=3)
model.set_observation_model(reader.knowledge_data.values)
params, history = model.fit([session['semantic_attribute_seq']])

# 3. Generate visualizations
plotter = AspectPlotter()
plotter.create_comprehensive_report({
    'sessions': [session],
    'models': [model],
    'params': [params],
    'history': [history]
})
```

### Command Line Testing
```bash
# Test with real data files
python test_with_arguments.py \
  --knowledge sample_data/knowledge.csv \
  --gaze sample_data/170116024150092.csv \
  --layout sample_data/170116024150092.list

# Test visualization pipeline
python test_visualization.py

# Test complete 4-layer pLSA
python test_four_layer.py

# Test data processing pipeline
python test_actual_data.py
```

## Data Format

### Required Files

1. **knowledge.csv**: Feature→Observation mapping matrix
   ```
   Items × Attributes matrix (e.g., 12×5)
   Values: Integer attribute levels (1-3)
   Purpose: Defines fixed P(o|v) layer
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
5. **Semantic Conversion**: Generate feature vectors via knowledge base
6. **Session Assembly**: Create temporal sequences for pLSA training

## Visualization Capabilities

### Complete MATLAB plot_*.m Equivalence

The visualization system provides full translation of all MATLAB plotting functions:

#### 1. Likelihood Analysis (`plot_L_topic.m`)
- Compare model performance across different topic numbers
- Training convergence monitoring
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
- Feature-topic relationship visualization
- Probability distribution heat maps
- Topic interpretation support

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
- Circular correlation matrix visualization
- Feature-aspect relationship mapping
- Custom color schemes (red-white-blue)
- Matrix subset analysis

### Sample Visualization Code
```python
from src.visualization.plotter import AspectPlotter

plotter = AspectPlotter()

# Individual plot types
plotter.plot_likelihood_comparison(models, histories)
plotter.plot_confusion_matrix(true_labels, predicted_labels)
plotter.plot_duration_by_user(session_data)
plotter.plot_aspect_matrices(params['vz'])
plotter.plot_similarity_analysis(session_data)
plotter.plot_evaluation_metrics(metrics_data)

# New additional plot types
plotter.plot_original_gaze_trajectories(gaze_data)
plotter.plot_aspect_correlation_circles(aspect_matrix)

# Comprehensive report generation
plotter.create_comprehensive_report(experimental_data)
```

## Testing

### Available Test Suite

```bash
# Core functionality
python test_actual_data.py          # Real data processing pipeline
python test_four_layer.py           # 4-layer pLSA training validation
python test_gaze_pipeline.py        # Complete data processing workflow

# Visualization system  
python test_visualization.py        # All plotting functions
python test_additional_plots.py     # Extended plotting functions
python test_with_arguments.py       # Command-line interface

# Integration tests
python -m src.data.gaze_reader      # Data reader demonstration
python -m src.models.plsa           # Model training demonstration
```

### Sample Test Output

```
Testing 4-layer pLSA with real data:
✅ Loaded knowledge base: (12, 5)
✅ Session loaded: 56 timepoints
✅ Training model with 3 topics...
✅ Training completed: 3 iterations
✅ Final likelihood: -315.4202
✅ Generated visualizations: 8 plots
✅ Comprehensive report saved
```

## Project Structure

```
python-ver/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── gaze_reader.py          # LoadGaze.m + ReadwithTimestamp.m
│   ├── models/
│   │   ├── __init__.py  
│   │   └── plsa.py                 # 4-layer pLSA implementation
│   ├── utils/
│   │   ├── __init__.py
│   │   └── math_utils.py           # Mathematical utilities
│   └── visualization/
│       ├── __init__.py
│       └── plotter.py              # Complete plot_*.m series
├── figures/                        # Generated visualizations
├── results/                        # Model outputs (2-layer legacy)
├── results_four_layer/             # 4-layer model outputs
├── sample_data/                    # Example experimental data
├── aspectmodel/                    # Virtual environment
├── requirements.txt                # Python dependencies
├── README.md                       # This documentation
└── test_*.py                       # Comprehensive test suite
```

## Key Achievements

### ✅ Complete MATLAB Translation
- **Exact Algorithm Equivalence**: All mathematical operations replicate MATLAB
- **Data Processing**: LoadGaze.m and ReadwithTimestamp.m fully translated
- **Model Architecture**: Correct 4-layer Session→Topic→Feature→Observation
- **Visualization**: All plot_*.m functions converted with enhanced styling including gaze trajectories and correlation circles

### ✅ Enhanced Functionality  
- **Type Safety**: Full type hints and validation
- **Error Handling**: Robust error checking and recovery
- **Modular Design**: Clean separation of concerns
- **Comprehensive Testing**: Full test suite with real data validation
- **Modern Visualization**: matplotlib/seaborn with publication-quality output

### ✅ Real Data Integration
- **Experimental Data**: Processes actual eye-tracking recordings
- **Multiple Formats**: Handles .csv, .list, knowledge base files
- **Batch Processing**: Support for multiple sessions and participants
- **Statistical Analysis**: Complete evaluation metrics and comparisons

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
- **Features**: Handles 5-50 feature dimensions
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
   # Verify data preprocessing
   python test_gaze_pipeline.py
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
