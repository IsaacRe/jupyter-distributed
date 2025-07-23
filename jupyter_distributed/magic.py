"""
Distributed Magic Commands for Jupyter with Persistent Processes

This module provides the %distribute magic command that allows parallel execution
of cells across multiple, persistent processes, maintaining variable state
between cell executions.
"""

import sys
import time
import multiprocess as mp
import threading
import queue
from typing import Dict, Any, List, Optional
from IPython.core.magic import Magics, line_cell_magic, magics_class
from IPython.core.magic_arguments import (argument, magic_arguments, parse_argstring)
import dill

# Worker process function
def worker_process(input_queue, output_queue, process_id):
    """
    A persistent worker process that maintains its own namespace and executes code.
    """
    namespace = {'__process_id__': process_id}
    
    # Custom stdout class that streams to queue with prefix
    class StreamingStdout:
        def __init__(self, process_id, output_queue):
            self.process_id = process_id
            self.output_queue = output_queue
            self.buffer = ""
        
        def write(self, text):
            self.buffer += text
            while '\n' in self.buffer:
                line, self.buffer = self.buffer.split('\n', 1)
                if line:
                    self.output_queue.put({'type': 'stdout', 'data': f"[Process {self.process_id}] {line}"})
        
        def flush(self):
            if self.buffer.strip():
                self.output_queue.put({'type': 'stdout', 'data': f"[Process {self.process_id}] {self.buffer.rstrip()}"})
                self.buffer = ""

    streaming_stdout = StreamingStdout(process_id, output_queue)
    original_stdout = sys.stdout
    sys.stdout = streaming_stdout

    while True:
        try:
            task = input_queue.get()
            if task is None:  # Shutdown signal
                break

            code = task.get('code')
            sync_vars = task.get('sync_vars')

            if sync_vars:
                namespace.update(sync_vars)

            try:
                exec(code, namespace)
                streaming_stdout.flush()
                
                # Filter out unpicklable objects from the namespace
                result_vars = {}
                for key, value in namespace.items():
                    if not key.startswith('__'):
                        try:
                            dill.dumps(value)
                            result_vars[key] = value
                        except (TypeError, AttributeError, dill.PicklingError):
                            pass # Skip unpicklable
                
                output_queue.put({'type': 'result', 'process_id': process_id, 'vars': result_vars, 'error': None})

            except Exception as e:
                streaming_stdout.flush()
                output_queue.put({'type': 'result', 'process_id': process_id, 'vars': {}, 'error': str(e)})

        except (KeyboardInterrupt, EOFError):
            break
    
    sys.stdout = original_stdout


@magics_class
class DistributedMagics(Magics):
    """Magic commands for distributed parallel execution with persistent processes."""
    
    def __init__(self, shell=None):
        super().__init__(shell)
        self.workers = {}  # {n_processes: [worker_info]}

    def _get_or_create_workers(self, n_processes: int):
        """Get or create a set of persistent worker processes."""
        if n_processes not in self.workers:
            self.workers[n_processes] = []
            for i in range(n_processes):
                input_queue = mp.Queue()
                output_queue = mp.Queue()
                process = mp.Process(target=worker_process, args=(input_queue, output_queue, i), daemon=True)
                process.start()
                self.workers[n_processes].append({
                    'process': process,
                    'input_queue': input_queue,
                    'output_queue': output_queue,
                    'namespace': {}
                })
        return self.workers[n_processes]

    @line_cell_magic
    @magic_arguments()
    @argument('n_processes', type=int, help='Number of processes to distribute across')
    @argument('--sync', action='store_true', help='Synchronize variables from process 0 to all others after execution')
    @argument('--timeout', type=int, default=None, help='Timeout in seconds for execution')
    def distribute(self, line, cell=None):
        """
        Distribute cell execution across n persistent processes.
        
        Usage:
            %distribute n [--sync] [--timeout SECONDS]
            %%distribute n [--sync] [--timeout SECONDS]
            <cell content>
        
        Variables persist in each process across multiple calls.
        """
        args = parse_argstring(self.distribute, line)
        n_processes = args.n_processes
        
        if n_processes <= 0:
            print("Error: Number of processes must be positive")
            return
        
        code = cell if cell is not None else ""
        if not code.strip():
            print("Error: No code to execute")
            return
            
        workers = self._get_or_create_workers(n_processes)
        
        print(f"Distributing execution across {n_processes} persistent processes...")
        start_time = time.time()

        # Send code to all worker processes
        for worker in workers:
            worker['input_queue'].put({'code': code})

        # Collect results and stream output
        results = [None] * n_processes
        errors = []
        completed_count = 0
        
        # Timeout handling
        timeout_event = threading.Event()
        if args.timeout:
            timer = threading.Timer(args.timeout, timeout_event.set)
            timer.start()

        while completed_count < n_processes and not timeout_event.is_set():
            for i, worker in enumerate(workers):
                if results[i] is not None:
                    continue
                try:
                    output = worker['output_queue'].get(timeout=0.01)
                    
                    if output['type'] == 'stdout':
                        print(output['data'], flush=True)
                    elif output['type'] == 'result':
                        results[i] = output
                        completed_count += 1
                        if output['error']:
                            errors.append(f"Process {output['process_id']}: {output['error']}")
                        else:
                            # Update local cache of namespace
                            worker['namespace'].update(output['vars'])

                except queue.Empty:
                    continue
        
        if args.timeout:
            timer.cancel()

        if timeout_event.is_set():
            print(f"Execution timed out after {args.timeout} seconds.")
            # Terminate and restart workers on timeout to clear state
            self._cleanup_workers(n_processes)
            return

        execution_time = time.time() - start_time
        
        if errors:
            print(f"Execution completed with errors in {len(errors)} processes:")
            for error in errors:
                print(f"  {error}")
        else:
            print(f"Successfully executed in all {n_processes} processes")
            
        print(f"Execution time: {execution_time:.2f} seconds")

        # Synchronize variables if requested
        if args.sync and not errors:
            self._sync_variables(n_processes)
            print("Variables synchronized from process 0 to all others.")

    def _sync_variables(self, n_processes: int):
        """Synchronize variables from process 0 to all other processes."""
        workers = self.workers.get(n_processes)
        if not workers or not workers[0]['namespace']:
            return

        # Get all picklable variables from process 0's namespace
        source_vars = {}
        for key, value in workers[0]['namespace'].items():
            try:
                dill.dumps(value)
                source_vars[key] = value
            except (TypeError, AttributeError, dill.PicklingError):
                pass

        # Send these variables to all other processes for synchronization
        for i in range(1, n_processes):
            workers[i]['input_queue'].put({'code': '', 'sync_vars': source_vars})
            workers[i]['namespace'].update(source_vars) # Update local cache

    def _cleanup_workers(self, n_processes: Optional[int] = None):
        """Terminate and clean up worker processes."""
        procs_to_clean = self.workers.items() if n_processes is None else [(n_processes, self.workers.get(n_processes, []))]
        
        for n, worker_list in procs_to_clean:
            for worker in worker_list:
                try:
                    worker['input_queue'].put(None) # Signal shutdown
                    worker['process'].join(timeout=1.0)
                    if worker['process'].is_alive():
                        worker['process'].terminate()
                except Exception:
                    pass
        
        if n_processes is None:
            self.workers.clear()
        elif n_processes in self.workers:
            del self.workers[n_processes]

    def __del__(self):
        """Clean up all worker processes when the magic object is destroyed."""
        self._cleanup_workers()

# Standalone function for direct registration
def distribute_magic(line, cell=None):
    """Standalone distribute magic function."""
    # This is not ideal for persistence; a single instance should be used.
    # In a real IPython extension, you would register the class.
    if not hasattr(distribute_magic, '_instance'):
        distribute_magic._instance = DistributedMagics()
    return distribute_magic._instance.distribute(line, cell)
