# DiBS: Differentiable Bayesian Structure Learning

This repository implements DiBS (Differentiable Bayesian Structure Learning).

## Project Structure

```
.
├── data/
│   └── graph_data.py          # Data generation utilities
|──debug/
      -debug_notebook.ipynb experiments and changes made to the code will be done here
      -dibs_debug.py same dibs model to be changed and tested for the debugging
├── models/
│   ├── __init__.py
│   ├── dibs.py               # Core DiBS implementation
│   └── utils.py              # Utility functions
├── notebooks/
│   └── dibs_experiment.ipynb # Main experiment notebook (use this)
├── tests/
│   └── 
├── main.py                   # empty
├── environment.yml           # Conda environment specification
├── requirements.txt          # Pip requirements
└── README.md                 # This file
```

## Setup Instructions

### Option 1: Using Conda (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd dibs_clean
   ```

2. **Create and activate conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate dibs_env
   ```



## Usage

### Running Experiments

1. **Start Jupyter Lab**
   ```bash
   jupyter lab
   ```

2. **Open and run the experiment notebook**
   - Navigate to `notebooks/dibs_experiment.ipynb`
   - Follow the cells to run DiBS experiments

### Configuration

The main configuration is handled through the `Config` class in the notebook. Key parameters include:

- `d_nodes`: Number of nodes in the graph
- `n_samples`: Number of data samples
- `alpha_val`, `beta_val`: DiBS hyperparameters
- `lr`: Learning rate
- `num_iterations`: Training iterations

### MLflow Tracking

Experiments are automatically tracked using MLflow. To view results:

```bash
mlflow ui
```

Then navigate to `http://localhost:5000` in your browser.

## Key Components

### Data Generation
- **Simple Chain**: Creates a simple chain graph structure
- **Synthetic**: Generates random DAGs with specified parameters

### Models
- **DiBS Core**: Implementation of the DiBS algorithm
- **Gradient Computation**: Stable gradient computation with logsumexp problematic
- **Constraint Handling**: Acyclicity constraints

### Training Loops
- **Basic Training**: Simple gradient-based optimization
- **Enhanced Training**: Advanced training with monitoring and MLflow integration

## Testing NOT YET DONE

Run tests using pytest: 
```bash
pytest tests/
```

## Dependencies

Core dependencies:
- PyTorch: Deep learning framework
- NumPy: Numerical computing
- MLflow: Experiment tracking
- NetworkX: Graph analysis
- python-igraph: Graph processing
- Matplotlib: Visualization

## Troubleshooting

### Known Issues

- **LogSumExp Stability**: correct logsumexp is needed for gradient calculations are problematic
- **Gradient theta and z update**:  both should be updated, the gradients are incorrect, 
- **Positive log joint**: the log joint gives positive and very big values which it shuldnt have. Should test with a basic model the issues only wit related to likelihood function.
