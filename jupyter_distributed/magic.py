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
from datetime import datetime
import os
from typing import Dict, Any, List, Optional
from IPython.core.magic import Magics, line_cell_magic, magics_class
from IPython.core.magic_arguments import (argument, magic_arguments, parse_argstring)
import dill

from .utils.ast import find_undefined_variables

# Worker process function
def worker_process(input_queue, output_queue, process_id):
    """
    A persistent worker process that maintains its own namespace and executes code.
    """
    namespace = {'__process_id__': process_id}
    
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

    # Diagnostic logging function
    diag_start = datetime.now()
    debug = False
    def log_diagnostic(message):
        if debug:
            elapsed_seconds = (datetime.now() - diag_start).total_seconds()
            elapsed_minutes = elapsed_seconds // 60
            elapsed_seconds %= 60
            relative_timestamp = f"{elapsed_minutes:.0f}:{elapsed_seconds:.5f}"
            sys.stdout.write(f"[DIAGNOSTIC] Process {process_id} at {relative_timestamp}: {message}\n")
            sys.stdout.flush()

    log_diagnostic(f"Worker process started with PID {os.getpid()}")

    while True:
        try:
            diag_start = datetime.now()
            log_diagnostic("Waiting for task from input queue")
            task = input_queue.get()
            debug = task.get('debug', False)

            if task is None:  # Shutdown signal
                log_diagnostic("Received shutdown signal")
                break

            request_type = task.get('task', 'execute')

            log_diagnostic("Received task, starting execution")
            code = task.get('code')
            vars_to_send = task.get('vars')
            namespace_to_load = task.get('namespace', {})

            try:
                warnings = []
                result_vars = {}
                
                if request_type == 'execute':
                    if code is None:
                        raise ValueError("no code provided for execution")
                    namespace.update(namespace_to_load)
                    log_diagnostic("About to execute code")
                    exec_start = time.time()
                    exec(code, namespace)
                    exec_end = time.time()
                    log_diagnostic(f"Code execution completed in {exec_end - exec_start:.3f} seconds")
                    sys.stdout.flush()

                    # Alert if stdout/stderr has been tampered with by the executed code
                    if not (isinstance(sys.stdout, QueueWriter) and isinstance(sys.stderr, QueueWriter)):
                        warnings.append("stdout/stderr have been modified by the executed code, which may lead to unexpected behavior.")
                elif request_type == 'load_vars':
                    log_diagnostic("Loading variables from namespace")
                    if vars_to_send is None:
                        raise ValueError("no variables specified to load")
                    for var in vars_to_send:
                        if var not in namespace:
                            raise NameError(f"name '{var}' is not defined")
                        log_diagnostic(f"Attempting to pickle variable '{var}'. Pickling large variables may cause OOM.")
                        value = namespace[var]
                        try:
                            dill.dumps(value)
                            result_vars[var] = value
                        except (TypeError, AttributeError, dill.PicklingError):
                            raise TypeError(f"variable '{var}' failed to pickle and cannot be loaded across processes")
                
                log_diagnostic("Sending result to output queue")
                output_queue.put({
                    'type': 'result',
                    'task': task.get('task', 'execute'),
                    'process_id': process_id,
                    'vars': result_vars,
                    'names': list(namespace.keys()),
                    'error': None,
                    'warnings': warnings})
                log_diagnostic("Result sent successfully")

            except KeyboardInterrupt:
                sys.stdout.write(f"Received SIGINT for process {process_id}\n")
                sys.stdout.flush()
                output_queue.put({'type': 'result', 'task': task, 'process_id': process_id, 'vars': {}, 'error': "Execution interrupted"})
            except Exception as e:
                sys.stdout.flush()
                output_queue.put({'type': 'result', 'task': task, 'process_id': process_id, 'vars': {}, 'error': str(e)})
        
        except KeyboardInterrupt:
            sys.stdout.write(f"Received SIGINT for process {process_id}\n")
            sys.stdout.flush()
            output_queue.put({'type': 'result', 'task': task, 'process_id': process_id, 'vars': {}, 'error': "Execution interrupted"})
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

    def _get_or_create_workers(self, n_processes: int) -> List[Dict[str, Any]]:
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
    
    def _interrupt_workers(self, n_processes: int, process_id: Optional[int] = None):
        if n_processes not in self.workers:
            print(f"Failed to interrupt distributed workers. No workers found for launch with {n_processes} processes.")
            return
        for proc_id, worker_meta in enumerate(self.workers[n_processes]):
            if process_id is None or proc_id == process_id:
                os.kill(worker_meta['process'].pid, signal.SIGINT)

    @line_cell_magic
    @magic_arguments()
    @argument('n_processes', type=int, help='Number of processes to distribute across')
    @argument('--load_vars', nargs='+', type=str, default=[],
              help='List of variable names to load from the main namespace into each worker')
    @argument('--debug', action='store_true', help='Enable diagnostic logging in worker processes')
    def distribute(self, line, cell=None):
        """
        Distribute cell execution across n persistent processes.

        Variables persist in each process across multiple calls.

        Usage:
            %distribute n [--debug]
            %%distribute n [--debug]
            <cell content>
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

        diag_start = datetime.now()
        def log_diagnostic(message):
            if args.debug:
                elapsed_seconds = (datetime.now() - diag_start).total_seconds()
                elapsed_minutes = elapsed_seconds // 60
                elapsed_seconds %= 60
                relative_timestamp = f"{elapsed_minutes:.0f}:{elapsed_seconds:.5f}"
                print(f"[MAIN] at {relative_timestamp}: {message}\n")

        # Get variables to load from the main namespace
        if args.load_vars:
            for var in args.load_vars:
                if var not in self.shell.user_ns:
                    raise NameError(f"name '{var}' is not defined in the main namespace")
            send_namespace_by_proc = {
                i: {var: self.shell.user_ns[var] for var in args.load_vars}
                for i in range(n_processes)
            }
        else:
            var_list_by_proc = self.get_main_vars_to_distribute(
                pool_id=n_processes,  # Use the smallest pool ID
                process_ids=list(range(n_processes)),
                cell=cell,
            )
            send_namespace_by_proc = {
                proc_id: {var: self.shell.user_ns[var] for var in var_list}
                for proc_id, var_list in var_list_by_proc.items()
            }

        # Send code to all worker processes
        log_diagnostic(f"Starting to send tasks to {n_processes} workers")
        for i, worker in enumerate(workers):
            # Clear output queue before starting
            while not worker['output_queue'].empty():
                worker['output_queue'].get()
            worker['input_queue'].put({'type': 'execute', 'code': code, 'debug': args.debug, 'namespace': send_namespace_by_proc[i], 'task': 'execute'})
            log_diagnostic(f"Task sent to worker {i}")

        # Collect results and stream output
        results = [None] * n_processes
        errors = []
        completed_count = 0

        log_diagnostic(f"Starting result collection")
        last_activity = time.time()
        
        try:
            while completed_count < n_processes:
                activity_this_cycle = False
                for i, worker in enumerate(workers):
                    if results[i] is not None:
                        continue
                    try:
                        output = worker['output_queue'].get(timeout=0.01)
                        activity_this_cycle = True
                        last_activity = time.time()

                        if output['type'] == 'stdout':
                            carriage_return = '\r'
                            print(f"[Process {output['process_id']}] {output['data'].strip(carriage_return)}", flush=True)
                        elif output['type'] == 'result' and output['task'] == 'execute':
                            log_diagnostic(f"Received result from worker {i} with process ID {output['process_id']}")
                            results[i] = output
                            completed_count += 1
                            if output['error']:
                                errors.append(f"Process {output['process_id']}: {output['error']}")
                            else:
                                # Update local cache of namespace
                                if output.get('names') is not None:
                                    worker['namespace'] = output['names']
                        if output.get('warnings'):
                            for warning in output['warnings']:
                                print(f"[Process {output['process_id']}] Warning: {warning}", flush=True)
                    except queue.Empty:
                        continue
                
                # Log if we haven't seen activity for a while
                if not activity_this_cycle and time.time() - last_activity > 5.0:
                    print(f"[MAIN] No activity for {time.time() - last_activity:.1f}s, completed: {completed_count}/{n_processes}")
                    last_activity = time.time()  # Reset to avoid spam
        
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
                        output = worker['output_queue'].get(timeout=1.0)
                        if output['type'] == 'result' and output['error'] == 'Execution interrupted':
                            break
                except queue.Empty:
                    print(f"Process {i} did not respond to interrupt after 1 seconds.")
                    continue
                print(f"Process {i} interrupted successfully.")
        
        execution_time = time.time() - start_time
            
        print(f"Execution time: {execution_time:.2f} seconds")

    @line_cell_magic
    @magic_arguments()
    @argument('vars', nargs='*', type=str, help='List of variable names to save')
    @argument('--from', type=int, default=0, dest='process_id', help='Process ID to load variables from')
    @argument('--debug', action='store_true', help='Enable diagnostic logging in worker processes')
    @argument('--noexec', action='store_false', dest='exec_code', default=True,
              help='Do not execute the cell code, only load variables')
    def load_vars(self, line: str, cell: Optional[str] = None):
        """
        Load variables from a specific worker process and update the current
        namespace with their values.

        If no variables are explicitly specified, it will attempt to extract
        undefined variables from the current cell's code.
        
        Usage:
            %load_vars VAR1 VAR2 ... [--from PROCESS_ID] [--debug]
            %%load_vars [VAR1 VAR2 ...] [--from PROCESS_ID] [--debug]
            <cell content>
        """
        diag_start = datetime.now()
        def log_diagnostic(message):
            if args.debug:
                elapsed_seconds = (datetime.now() - diag_start).total_seconds()
                elapsed_minutes = elapsed_seconds // 60
                elapsed_seconds %= 60
                relative_timestamp = f"{elapsed_minutes:.0f}:{elapsed_seconds:.5f}"
                print(f"[MAIN] at {relative_timestamp}: {message}\n")
        
        args = parse_argstring(self.load_vars, line)
        self.run_load_vars(
            var_list=args.vars,
            process_id=args.process_id,
            debug=args.debug,
            cell=cell,
        )

        # if we've loaded variables successfully execute the cell code with the modified scope
        if args.exec_code and cell is not None:
            log_diagnostic("Executing cell code with loaded variables")
            self.shell.run_cell(cell, store_history=False, silent=False, shell_futures=True)

    def run_load_vars(
        self,
        var_list: List[str],
        process_id: int,
        debug: bool = False,
        cell: Optional[str] = None,
    ):      
        # Find the process pool with the smallest number of distributed processes
        if not self.workers:
            raise RuntimeError("No distributed workers available. Run %%distribute first to create worker processes.")
        
        pool_id = min(self.workers.keys())
        
        # Validate process_id exists in the selected pool
        if process_id >= len(self.workers[pool_id]):
            raise ValueError(f"Process ID {process_id} does not exist in worker pool '{pool_id}'. "
                             f"Available process IDs: 0-{len(self.workers[pool_id])-1}")
        
        worker = self.workers[pool_id][process_id]

        if var_list:
            pass
        elif cell is None:
            raise ValueError("no variables specified to load. Use `%load_vars <name1> <name2> ...` to specify variable names "
                             "or run with `%%load_vars` so the current cell's code is available to infer variables.")
        else:
            undef_vars = find_undefined_variables(cell)
            var_list = (undef_vars - set(self.shell.user_ns.keys())).intersection(worker['namespace'])
            if mutually_defined := undef_vars.intersection(self.shell.user_ns.keys()).intersection(worker['namespace']):
                mutual_string = ", ".join(list(map(lambda x: f"'{x}'", mutually_defined))[:2])
                mutual_string += f", +{len(mutually_defined) - 2} more" if len(mutually_defined) > 2 else ""
                print(f"Warning: the following names are mutually defined in main namespace and distributed namespaces: "
                      f"{mutual_string}. To load from a distributed namespace use `%%load_vars <name1> <name2> ...`, "
                      "otherwise values from the main namespace will be used.")
        
        if not var_list:
            return

        diag_start = datetime.now()
        def log_diagnostic(message):
            if debug:
                elapsed_seconds = (datetime.now() - diag_start).total_seconds()
                elapsed_minutes = elapsed_seconds // 60
                elapsed_seconds %= 60
                relative_timestamp = f"{elapsed_minutes:.0f}:{elapsed_seconds:.5f}"
                print(f"[MAIN] at {relative_timestamp}: {message}\n")

        log_diagnostic(f"Loading variables to main namespace from process {process_id}: {var_list}")

        # query distributed process for variables
        worker['input_queue'].put({'type': 'request', 'task': 'load_vars', 'vars': var_list, 'debug': debug})

        try:
            while True:
                try:
                    output = worker['output_queue'].get(timeout=0.01)

                    if output['type'] == 'stdout':
                        carriage_return = '\r'
                        print(f"[Process {output['process_id']}] {output['data'].strip(carriage_return)}", flush=True)
                    elif output['type'] == 'result' and output['task'] == 'load_vars':
                        log_diagnostic(f"Received result from worker {process_id} with process ID {output['process_id']}")
                        if output['error']:
                            print(f"Execution error in process {output['process_id']}: {output['error']}", flush=True)
                        if output.get('warnings'):
                            for warning in output['warnings']:
                                print(f"[Process {output['process_id']}] Warning: {warning}", flush=True)
                        if output['vars']:
                            # Update local cache of namespace
                            self.shell.user_ns.update(output['vars'])
                            return
                except queue.Empty:
                    continue

        except KeyboardInterrupt:
            print("\nExecution interrupted by user. Sending interrupt to workers...")
            self._interrupt_workers(pool_id, process_id=process_id)
            try:
                while True:
                    output = worker['output_queue'].get(timeout=1.0)
                    if output['type'] == 'result' and output['error'] == 'Execution interrupted':
                        break
            except queue.Empty:
                print(f"Process {process_id} did not respond to interrupt after 1 seconds.")
                return
            print(f"Process {process_id} interrupted successfully.")

    def get_main_vars_to_distribute(
        self,
        pool_id: str = "default",
        process_ids: List[int] = [],
        cell: Optional[str] = None,
    ) -> Dict[int, str]:
        # Find the process pool with the smallest number of distributed processes
        if not self.workers:
            raise RuntimeError("No distributed workers available. Run %%distribute first to create worker processes.")
        
        if not process_ids:
            process_ids = list(range(len(self.workers[pool_id])))

        pool_id = min(self.workers.keys())
        
        # Validate process_id exists in the selected pool
        for process_id in process_ids:
            if process_id >= len(self.workers[pool_id]):
                raise ValueError(f"Process ID {process_id} does not exist in worker pool '{pool_id}'. "
                                 f"Available process IDs: 0-{len(self.workers[pool_id])-1}")

        var_list_by_process = {}
        if cell is None:
            raise ValueError("no variables specified to load. Use `%load_vars <name1> <name2> ...` to specify variable names "
                             "or run with `%%load_vars` so the current cell's code is available to infer variables.")
        else:
            undef_vars = find_undefined_variables(cell)
            for process_id in process_ids:
                worker = self.workers[pool_id][process_id]
                var_list = (undef_vars - set(worker['namespace'])).intersection(self.shell.user_ns.keys())
                if mutually_defined := undef_vars.intersection(set(worker['namespace'])).intersection(self.shell.user_ns.keys()):
                    mutual_string = ", ".join(list(map(lambda x: f"'{x}'", mutually_defined))[:2])
                    mutual_string += f", +{len(mutually_defined) - 2} more" if len(mutually_defined) > 2 else ""
                    print(f"Warning: the following names are mutually defined in main namespace and process {process_id}: "
                          f"{mutual_string}. To load from the main namespace use `%%load_vars <name1> <name2> ...`, "
                          f"otherwise values from the namespace of process {process_id} will be used.")
                var_list_by_process[process_id] = var_list
        
        return var_list_by_process

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

    def pre_cell_hook(self, info):
        """A hook that runs before each cell is executed."""
        code = info.raw_cell
        if (code.startswith("%%distribute") or
            code.startswith("%%load_vars")):  # skip for distributed calls
            return
        self.run_load_vars(
            var_list=[],      # infer variables to load from cell content
            process_id=0,
            debug=True,
            cell=code,
        )


# Standalone function for direct registration
def distribute_magic(line, cell=None):
    """Standalone distribute magic function."""
    # This is not ideal for persistence; a single instance should be used.
    # In a real IPython extension, you would register the class.
    if not hasattr(distribute_magic, '_instance'):
        distribute_magic._instance = DistributedMagics()
    return distribute_magic._instance.distribute(line, cell)
