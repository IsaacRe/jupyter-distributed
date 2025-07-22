# Quick Start Guide - Jupyter Distributed Extension

## Installation

The extension has been installed in development mode. To use it in a Jupyter notebook:

## Basic Usage

1. **Start Jupyter Notebook or JupyterLab:**
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

2. **Load the extension in your notebook:**
   ```python
   %load_ext jupyter_distributed
   ```

3. **Use the `%distribute` magic command:**

   ### Simple parallel execution:
   ```python
   %%distribute 4
   import os
   import time
   
   # Each process gets a unique __process_id__
   process_id = __process_id__
   result = f"Hello from process {process_id}! PID: {os.getpid()}"
   print(result)
   
   # Simulate some work
   time.sleep(1)
   computation_result = process_id ** 2
   ```

   ### Variable persistence:
   ```python
   %%distribute 4
   # Variables from previous cells are still available
   print(f"Process {__process_id__}: Previous result was {computation_result}")
   new_result = computation_result * 10
   ```

   ### Synchronization:
   ```python
   %%distribute 4 --sync
   # Only process 0 sets this variable
   if __process_id__ == 0:
       shared_data = "This will be shared across all processes"
       shared_number = 42
   
   print(f"Process {__process_id__} finished")
   ```

   ### With timeout:
   ```python
   %%distribute 2 --timeout 10
   import time
   
   print(f"Process {__process_id__} starting work...")
   time.sleep(2)  # Will complete within timeout
   print(f"Process {__process_id__} completed work!")
   ```

## Command Options

- `n`: Number of processes (required)
- `--sync`: Synchronize variables from process 0 to all others
- `--timeout SECONDS`: Set maximum execution time

## Special Variables

- `__process_id__`: Unique identifier (0 to n-1) for each process

## Examples

Check the `examples/basic_usage.ipynb` notebook for more detailed examples including:
- Parallel data processing
- Monte Carlo simulations
- Machine learning model training
- Error handling

## Testing

Run the demo script to test basic functionality:
```bash
python demo.py
```

Run the test suite:
```bash
python -m pytest tests/ -v
```

## Troubleshooting

1. **Extension not loading**: Make sure you've installed the package and restarted your Jupyter kernel
2. **Import errors**: Ensure all dependencies are installed (`multiprocess`, `dill`, etc.)
3. **Process errors**: Check that your code is serializable and doesn't use non-picklable objects
4. **Performance issues**: Consider the overhead of process creation for small computations

## Next Steps

- Try the examples in `examples/basic_usage.ipynb`
- Experiment with different numbers of processes
- Test with your own computational workloads
- Explore the synchronization features for coordinated parallel work
