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
import signal
import os
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

    # Timeout handler
    class TimeoutException(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutException()

    signal.signal(signal.SIGALRM, timeout_handler)
    
    # Redirect stdout to a queue
    class QueueWriter:
        def __init__(self, queue, process_id):
            self.queue = queue
            self.process_id = process_id

        def write(self, text):
            self.queue.put({'type': 'stdout', 'process_id': self.process_id, 'data': text})

        def flush(self):
            pass

    sys.stdout = QueueWriter(output_queue, process_id)
    sys.stderr = QueueWriter(output_queue, process_id)

    sys.stdout.write(f"Worker process {process_id} started with PID {os.getpid()}\n")
    sys.stdout.flush()

    while True:
        try:
            task = input_queue.get()
            if task is None:  # Shutdown signal
                break

            code = task.get('code')
            timeout = task.get('timeout')

            try:
                if timeout:
                    signal.alarm(timeout)
                
                exec(code, namespace)
                sys.stdout.flush()
                
                if timeout:
                    signal.alarm(0) # Cancel alarm
                
                # Filter out unpicklable objects from the namespace
                result_vars = {}
                for key, value in namespace.items():
                    if not key.startswith('__'):
                        try:
                            dill.dumps(value)
                            result_vars[key] = value
                        except (TypeError, AttributeError, dill.PicklingError):
                            pass # Skip unpicklable
                
                # For now do not return variables from distributed cells to non-distributed runtime
                output_queue.put({'type': 'result', 'process_id': process_id, 'vars': {}, 'error': None})

            except TimeoutException:
                sys.stdout.flush()
                output_queue.put({'type': 'result', 'process_id': process_id, 'vars': {}, 'error': f"Execution timed out after {timeout} seconds"})
            except KeyboardInterrupt:
                sys.stdout.write(f"Received SIGINT for process {process_id}\n")
                sys.stdout.flush()
                output_queue.put({'type': 'result', 'process_id': process_id, 'vars': {}, 'error': "Execution interrupted"})
            except Exception as e:
                sys.stdout.flush()
                output_queue.put({'type': 'result', 'process_id': process_id, 'vars': {}, 'error': str(e)})
        
        except KeyboardInterrupt:
            sys.stdout.write(f"Received SIGINT for process {process_id}\n")
            sys.stdout.flush()
            output_queue.put({'type': 'result', 'process_id': process_id, 'vars': {}, 'error': "Execution interrupted"})
        except EOFError:
            break


@magics_class
class DistributedMagics(Magics):
    """Magic commands for distributed parallel execution with persistent processes."""
    
    def __init__(self, shell=None):
        super().__init__(shell)
        self.workers = {}  # {n_processes: [worker_info]}
        # Use 'spawn' start method for multiprocessing to avoid issues with
        # forked processes in interactive environments like Jupyter.
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # This can happen if the context is already set and we can't force it.
            # We'll proceed, but the user might see warnings.
            pass

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
    
    def _interrupt_workers(self, n_processes: int):
        if n_processes not in self.workers:
            print(f"Failed to interrupt distributed workers. No workers found for launch with {n_processes} processes.")
            return
        for worker_meta in self.workers[n_processes]:
            os.kill(worker_meta['process'].pid, signal.SIGINT)

    @line_cell_magic
    @magic_arguments()
    @argument('n_processes', type=int, help='Number of processes to distribute across')
    @argument('--timeout', type=int, default=None, help='Timeout in seconds for execution')
    def distribute(self, line, cell=None):
        """
        Distribute cell execution across n persistent processes.
        
        Usage:
            %distribute n [--timeout SECONDS]
            %%distribute n [--timeout SECONDS]
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
            # Clear output queue before starting
            while not worker['output_queue'].empty():
                worker['output_queue'].get()
            worker['input_queue'].put({'code': code, 'timeout': args.timeout})

        # Collect results and stream output
        results = [None] * n_processes
        errors = []
        completed_count = 0
        
        # Timeout handling
        timeout_event = threading.Event()
        if args.timeout:
            timer = threading.Timer(args.timeout, timeout_event.set)
            timer.start()

        try:
            while completed_count < n_processes and not timeout_event.is_set():
                for i, worker in enumerate(workers):
                    if results[i] is not None:
                        continue
                    try:
                        output = worker['output_queue'].get(timeout=0.01)

                        if output['type'] == 'stdout':
                            carriage_return = '\r'
                            print(f"[Process {output['process_id']}] {output['data'].strip(carriage_return)}", flush=True)
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
        
            if errors:
                print(f"Execution completed with errors in {len(errors)} processes:")
                for error in errors:
                    print(f"  {error}")
            else:
                print(f"Successfully executed in all {n_processes} processes")

        except KeyboardInterrupt:
            print("\nExecution interrupted by user. Sending interrupt to workers...")
            self._interrupt_workers(n_processes)
            for i, worker in enumerate(workers):
                try:
                    while True:
                        output = worker['output_queue'].get(timeout=2.0)
                        if output['type'] == 'result' and output['error'] == 'Execution interrupted':
                            break
                except queue.Empty:
                    print(f"Process {i} did not respond to interrupt after 2 seconds.")
                    continue
                print(f"Process {i} interrupted successfully.")
        
        execution_time = time.time() - start_time
            
        print(f"Execution time: {execution_time:.2f} seconds")

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
