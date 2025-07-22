# Jupyter Distributed Extension

A Jupyter notebook extension that provides the `%distribute` magic command for parallel execution of cells across multiple processes while maintaining variable state between executions.

## Features

- **Parallel Execution**: Run notebook cells across multiple processes simultaneously
- **Variable Persistence**: Variables set in distributed cells are maintained across subsequent executions
- **Process Isolation**: Each process maintains its own namespace while sharing variables when needed
- **Synchronization**: Optional variable synchronization across all processes
- **Error Handling**: Robust error handling with detailed reporting per process
- **Timeout Support**: Set execution timeouts to prevent hanging processes
- **Easy Integration**: Simple magic command interface that integrates seamlessly with Jupyter

## Installation

### From Source

```bash
git clone https://github.com/yourusername/jupyter-distributed.git
cd jupyter-distributed
pip install -e .
```

### Using pip (when published)

```bash
pip install jupyter-distributed
```

## Usage

### Loading the Extension

```python
%load_ext jupyter_distributed
```

### Basic Usage

Use `%%distribute n` to run a cell across `n` processes:

```python
%%distribute 4
import os
import time

# Each process gets a unique __process_id__
process_id = __process_id__
result = f"Hello from process {process_id}! PID: {os.getpid()}"
print(result)

# Variables set here will be available in subsequent %distribute calls
computation_result = process_id ** 2
```

### Variable Persistence

Variables assigned in distributed cells are automatically preserved for subsequent executions:

```python
%%distribute 4
# Variables from previous cells are available
print(f"Process {__process_id__}: Previous result was {computation_result}")
new_result = computation_result * 10
```

### Synchronization

Use the `--sync` flag to synchronize variables across all processes after execution:

```python
%%distribute 4 --sync
# Only process 0 sets this variable
if __process_id__ == 0:
    shared_data = "This will be shared across all processes"
    shared_number = 42

print(f"Process {__process_id__} finished")
```

After synchronization, all processes will have access to variables set by process 0.

### Timeout Control

Set execution timeouts to prevent hanging processes:

```python
%%distribute 4 --timeout 10
import time

# This will timeout if it takes longer than 10 seconds
time.sleep(5)  # This will complete successfully
result = f"Process {__process_id__} completed"
```

## Command Reference

### `%distribute` / `%%distribute`

**Syntax:**
```
%distribute n [--sync] [--timeout SECONDS]
%%distribute n [--sync] [--timeout SECONDS]
<cell content>
```

**Parameters:**
- `n`: Number of processes to distribute execution across (required)
- `--sync`: Synchronize variables from process 0 to all other processes after execution
- `--timeout SECONDS`: Maximum execution time in seconds

**Special Variables:**
- `__process_id__`: Unique identifier (0 to n-1) for each process

## Examples

### Parallel Data Processing

```python
%%distribute 4
import numpy as np

# Different random seed for each process
np.random.seed(__process_id__)
data = np.random.randn(1000)

# Process data in parallel
mean_val = np.mean(data)
std_val = np.std(data)
processed_data = (data - mean_val) / std_val

print(f"Process {__process_id__}: Mean={mean_val:.3f}, Std={std_val:.3f}")
```

### Monte Carlo Simulation

```python
%%distribute 8
import numpy as np

def estimate_pi(n_samples):
    np.random.seed(__process_id__ * 1000)
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)
    inside_circle = (x**2 + y**2) <= 1
    return 4 * np.sum(inside_circle) / n_samples

# Each process estimates pi independently
pi_estimate = estimate_pi(100000)
print(f"Process {__process_id__}: π ≈ {pi_estimate:.6f}")
```

### Machine Learning Model Training

```python
%%distribute 4
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Different model for each process
models = {
    0: RandomForestClassifier(n_estimators=100),
    1: SVC(kernel='rbf'),
    2: LogisticRegression(max_iter=1000),
    3: KNeighborsClassifier(n_neighbors=5)
}

# Train assigned model
model = models[__process_id__]
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

print(f"Process {__process_id__}: Model accuracy = {accuracy:.4f}")
```

## How It Works

1. **Process Pool Management**: The extension maintains process pools for different sizes, reusing them across executions
2. **Variable Serialization**: Uses `dill` for robust serialization of Python objects between processes
3. **Namespace Isolation**: Each process maintains its own variable namespace, preventing conflicts
4. **AST Analysis**: Analyzes code to track variable assignments for persistence
5. **Error Isolation**: Errors in individual processes don't affect others

## Requirements

- Python 3.7+
- IPython 7.0+
- Jupyter
- multiprocess
- dill

## Limitations

- Variables must be serializable with `dill`
- Process creation overhead may not benefit small computations
- Memory usage scales with number of processes
- Not suitable for I/O bound tasks (use threading instead)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Changelog

### v0.1.0
- Initial release
- Basic `%distribute` magic command
- Variable persistence across executions
- Synchronization support
- Timeout control
- Error handling and reporting
