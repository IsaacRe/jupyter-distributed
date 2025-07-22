"""
Distributed Magic Commands for Jupyter

This module provides the %distribute magic command that allows parallel execution
of cells across multiple processes while maintaining variable state.
"""

import ast
import sys
import time
import pickle
import multiprocess as mp
import io
import contextlib
import threading
import queue
from typing import Dict, Any, List, Optional
from IPython.core.magic import Magics, line_cell_magic, magics_class
from IPython.core.magic_arguments import (argument, magic_arguments, parse_argstring)
from IPython.display import display, HTML
import dill


@magics_class
class DistributedMagics(Magics):
    """Magic commands for distributed parallel execution."""
    
    def __init__(self, shell=None):
        super().__init__(shell)
        self.process_pools = {}  # Store process pools by size
        self.process_namespaces = {}  # Store variable namespaces for each process
        self.active_processes = 0
        
    def _get_or_create_pool(self, n_processes: int):
        """Get or create a process pool of the specified size."""
        if n_processes not in self.process_pools:
            self.process_pools[n_processes] = mp.Pool(n_processes)
            self.process_namespaces[n_processes] = [{} for _ in range(n_processes)]
        return self.process_pools[n_processes], self.process_namespaces[n_processes]
    
    def _extract_assignments(self, code: str) -> List[str]:
        """Extract variable assignments from code to track what variables are set."""
        try:
            tree = ast.parse(code)
            assignments = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            assignments.append(target.id)
                        elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
                            for elt in target.elts:
                                if isinstance(elt, ast.Name):
                                    assignments.append(elt.id)
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    assignments.append(node.target.id)
                elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                    assignments.append(node.target.id)
            
            return assignments
        except SyntaxError:
            return []
    
    @staticmethod
    def _execute_in_process(args):
        """Execute code in a separate process with the given namespace."""
        code, namespace, process_id, output_queue = args
        
        # Create a clean execution environment with only picklable objects
        local_env = {}
        
        # Filter the namespace to only include picklable objects
        for key, value in namespace.items():
            try:
                # Test if the value can be pickled
                dill.dumps(value)
                local_env[key] = value
            except (TypeError, AttributeError, dill.PicklingError, Exception):
                # Skip unpicklable objects silently
                pass
        
        # Add the process ID
        local_env['__process_id__'] = process_id
        
        # Custom stdout class that streams to queue with prefix
        class StreamingStdout:
            def __init__(self, process_id, output_queue):
                self.process_id = process_id
                self.output_queue = output_queue
                self.buffer = ""
            
            def write(self, text):
                # Add text to buffer
                self.buffer += text
                
                # Process complete lines
                while '\n' in self.buffer:
                    line, self.buffer = self.buffer.split('\n', 1)
                    if line:  # Only send non-empty lines
                        self.output_queue.put(f"[Process {self.process_id}] {line}")
                
                # Handle case where text doesn't end with newline but we want immediate output
                if text and not text.endswith('\n') and '\n' not in text:
                    # For immediate output without newline, still send it
                    pass
            
            def flush(self):
                # Flush any remaining buffer content
                if self.buffer.strip():
                    self.output_queue.put(f"[Process {self.process_id}] {self.buffer.rstrip()}")
                    self.buffer = ""
        
        # Create streaming stdout
        streaming_stdout = StreamingStdout(process_id, output_queue)
        
        try:
            # Execute the code with streaming stdout
            original_stdout = sys.stdout
            sys.stdout = streaming_stdout
            
            exec(code, local_env)
            
            # Flush any remaining output
            streaming_stdout.flush()
            
            # Restore original stdout
            sys.stdout = original_stdout
            
            # Extract only the variables that were assigned in this execution
            # Filter out built-ins and special variables
            result_vars = {}
            for key, value in local_env.items():
                if (not key.startswith('__') or key == '__process_id__') and key not in ['In', 'Out', 'get_ipython']:
                    try:
                        # Test if the value can be pickled
                        dill.dumps(value)
                        result_vars[key] = value
                    except (TypeError, AttributeError, dill.PicklingError, Exception):
                        # Skip unpicklable objects
                        pass
            
            return process_id, result_vars, None
            
        except Exception as e:
            # Restore original stdout in case of error
            sys.stdout = original_stdout
            streaming_stdout.flush()
            return process_id, {}, str(e)
    
    @line_cell_magic
    @magic_arguments()
    @argument('n_processes', type=int, help='Number of processes to distribute across')
    @argument('--sync', action='store_true', help='Synchronize variables across all processes after execution')
    @argument('--timeout', type=int, default=None, help='Timeout in seconds for execution')
    def distribute(self, line, cell=None):
        """
        Distribute cell execution across n processes.
        
        Usage:
            %distribute n [--sync] [--timeout SECONDS]
            %%distribute n [--sync] [--timeout SECONDS]
            <cell content>
        
        Variables assigned in the cell will be available in subsequent %distribute calls
        with the same number of processes.
        """
        args = parse_argstring(DistributedMagics.distribute, line)
        n_processes = args.n_processes
        
        if n_processes <= 0:
            print("Error: Number of processes must be positive")
            return
        
        # Use cell content if this is a cell magic, otherwise use line content
        code = cell if cell is not None else ""
        
        if not code.strip():
            print("Error: No code to execute")
            return
        
        # Get or create process pool and namespaces
        pool, namespaces = self._get_or_create_pool(n_processes)
        
        # Extract variable assignments from the code
        assignments = self._extract_assignments(code)
        
        print(f"Distributing execution across {n_processes} processes...")
        start_time = time.time()
        
        # Create a shared output queue for streaming stdout
        output_queue = mp.Manager().Queue()
        
        # Prepare arguments for each process - filter namespaces for picklable objects only
        clean_namespaces = []
        for i in range(n_processes):
            clean_namespace = {}
            for key, value in namespaces[i].items():
                try:
                    # Test if the value can be pickled
                    dill.dumps(value)
                    clean_namespace[key] = value
                except (TypeError, AttributeError, dill.PicklingError, Exception):
                    # Skip unpicklable objects
                    pass
            clean_namespaces.append(clean_namespace)
        
        process_args = [(code, clean_namespaces[i], i, output_queue) for i in range(n_processes)]
        
        # Start a thread to handle streaming output
        output_thread_running = threading.Event()
        output_thread_running.set()
        
        def output_handler():
            """Handle streaming output from processes."""
            while output_thread_running.is_set():
                try:
                    # Get output from queue with timeout
                    output_line = output_queue.get(timeout=0.1)
                    print(output_line, flush=True)
                except:
                    # Timeout or empty queue, continue
                    continue
            
            # Process any remaining output after processes complete
            while not output_queue.empty():
                try:
                    output_line = output_queue.get_nowait()
                    print(output_line, flush=True)
                except:
                    break
        
        # Start output handling thread
        output_thread = threading.Thread(target=output_handler, daemon=True)
        output_thread.start()
        
        try:
            # Execute in parallel
            if args.timeout:
                results = pool.map_async(self._execute_in_process, process_args).get(timeout=args.timeout)
            else:
                results = pool.map(self._execute_in_process, process_args)
            
            # Stop output thread
            output_thread_running.clear()
            output_thread.join(timeout=1.0)  # Wait for output thread to finish
            
            execution_time = time.time() - start_time
            
            # Process results
            errors = []
            success_count = 0
            
            for process_id, result_vars, error in results:
                if error:
                    errors.append(f"Process {process_id}: {error}")
                else:
                    success_count += 1
                    # Update the namespace for this process
                    namespaces[process_id].update(result_vars)
            
            # Display results
            if errors:
                print(f"Execution completed with errors in {len(errors)} processes:")
                for error in errors:
                    print(f"  {error}")
            else:
                print(f"Successfully executed in all {n_processes} processes")
            
            print(f"Execution time: {execution_time:.2f} seconds")
            
            # Optionally sync variables across all processes
            if args.sync and success_count > 0:
                self._sync_variables(n_processes, assignments)
                print("Variables synchronized across all processes")
            
            # Display variable summary
            if assignments:
                print(f"Variables assigned: {', '.join(assignments)}")
                
        except mp.TimeoutError:
            print(f"Execution timed out after {args.timeout} seconds")
        except Exception as e:
            print(f"Error during parallel execution: {e}")
    
    def _sync_variables(self, n_processes: int, assignments: List[str]):
        """Synchronize specified variables across all processes."""
        if n_processes not in self.process_namespaces:
            return
        
        namespaces = self.process_namespaces[n_processes]
        
        # For each assignment, take the value from process 0 and copy to all others
        for var_name in assignments:
            if var_name in namespaces[0]:
                value = namespaces[0][var_name]
                for i in range(1, n_processes):
                    namespaces[i][var_name] = value
    
    def __del__(self):
        """Clean up process pools when the magic object is destroyed."""
        for pool in self.process_pools.values():
            try:
                pool.close()
                pool.join()
            except:
                pass


# Standalone function for direct registration
def distribute_magic(line, cell=None):
    """Standalone distribute magic function."""
    magic_instance = DistributedMagics()
    return magic_instance.distribute(line, cell)
