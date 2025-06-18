# DiBS: Differentiable Bayesian Structure Learning

This repository implements DiBS (Differentiable Bayesian Structure Learning).

## Project Structure

```
.
├── data/
│   └── graph_data.py          # Data generation utilities
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

3. **Install the package in development mode**
   ```bash
   pip install -e .
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
- **Gradient Computation**: Stable gradient computation with logsumexp
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

- **LogSumExp Stability**: correct logsumexp is needed for gradient calculations 
- **Gradient theta and z update**:  both should be updated, the gradients are incorrect, possibly due to logsumexp
- **Notes**: denominator for grad z log joint gumbell is always 1 

be careful when adding clamps  or limits  remove them first then add if needed
remove clamping errors and see if it does not work 

look the log weights in sum exp  it is possibly the reasoing for exploding/vanishing gradients
denominator is always 1  for the graz a and for both grad theta
fix the gradients  be carefull on gumbell and bernoulli(deterministic) soft gmat,  look where when to use

sampling definition(and the mc sampling loop structure and gumbell vs bernouilli) of the theta and z is little different due to reparametrization and the paper objectiev derivations given
grad z  should be wrt to grad z  expecation gumbell (1 gradient taken  of the expectation after sampling)
grad theta shoudl be wer the expectation then grad   (so take multiple gradients)

