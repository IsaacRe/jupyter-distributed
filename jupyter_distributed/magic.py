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
            vars_to_load = task.get('vars')

            try:
                warnings = []
                result_vars = {}
                
                if request_type == 'execute':
                    if code is None:
                        raise ValueError("no code provided for execution")
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
                    if vars_to_load is None:
                        raise ValueError("no variables specified to load")
                    for var in vars_to_load:
                        if var not in namespace:
                            raise NameError(f"name '{var}' is not defined")
                        log_diagnostic(f"Attempting to pickle variable '{var}'. Pickling large variables may cause OOM.")
                        value = namespace[var]
                        try:
                            dill.dumps(value)
                            result_vars[var] = value
                        except (TypeError, AttributeError, dill.PicklingError):
                            raise TypeError(f"variable '{var}' failed to pickle and cannot be loaded across processes")\
                
                log_diagnostic("Sending result to output queue")
                output_queue.put({'type': 'result', 'task': task, 'process_id': process_id, 'vars': result_vars, 'error': None, 'warnings': warnings})
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

        # Send code to all worker processes
        log_diagnostic(f"Starting to send tasks to {n_processes} workers")
        for i, worker in enumerate(workers):
            # Clear output queue before starting
            while not worker['output_queue'].empty():
                worker['output_queue'].get()
            worker['input_queue'].put({'type': 'execute', 'code': code, 'debug': args.debug})
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
                        elif output['type'] == 'result':
                            log_diagnostic(f"Received result from worker {i} with process ID {output['process_id']}")
                            results[i] = output
                            completed_count += 1
                            if output['error']:
                                errors.append(f"Process {output['process_id']}: {output['error']}")
                            else:
                                # Update local cache of namespace
                                worker['namespace'].update(output['vars'])
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
                    print(f"Process {i} did not respond to interrupt after 2 seconds.")
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
        
        worker_n_procs = min(self.workers.keys())
        
        # Validate process_id exists in the selected pool
        if process_id >= len(self.workers[worker_n_procs]):
            raise ValueError(f"Process ID {process_id} does not exist in the worker pool with {worker_n_procs} processes. "
                           f"Available process IDs: 0-{len(self.workers[worker_n_procs])-1}")
        
        worker = self.workers[worker_n_procs][process_id]

        if var_list:
            pass
        elif cell is None:
            raise ValueError("no variables specified to load. Use `%load_vars <name1> <name2> ...` to specify variable names "
                             "or run with `%%load_vars` so the current cell's code is available to infer variables.")
        else:
            var_list = find_undefined_variables(cell)
            if mutually_defined := var_list.intersection(self.shell.user_ns.keys()):
                mutual_string = ", ".join(list(map(lambda x: f"'{x}'", mutually_defined))[:2])
                mutual_string += f", +{len(mutually_defined) - 2} more" if len(mutually_defined) > 2 else ""
                abridged_var_list = ", ".join(list(mutually_defined)[:2])
                abridged_var_list += ", ..." if len(var_list) > 2 else ""
                name_names = "names" if len(mutually_defined) > 1 else "name"
                is_are = "are" if len(mutually_defined) > 1 else "is"
                it_them = "it" if len(mutually_defined) == 1 else "them"
                its_their_value_values = "its value" if len(mutually_defined) == 1 else "their values"
                print(f"Warning: {name_names} {mutual_string} {is_are} mutually defined in main namespace and distributed namespaces. "
                      f"To load {it_them} from a distributed namespace use `%%load_vars {abridged_var_list}`, otherwise "
                      f"{its_their_value_values} from the main namespace will be used.")
                var_list = list(var_list - set(mutually_defined))
            else:
                var_list = list(var_list)
        
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
        worker['input_queue'].put({'type': 'load_vars', 'task': 'load_vars', 'vars': var_list, 'debug': debug})

        try:
            while True:
                try:
                    output = worker['output_queue'].get(timeout=0.01)

                    if output['type'] == 'stdout':
                        carriage_return = '\r'
                        print(f"[Process {output['process_id']}] {output['data'].strip(carriage_return)}", flush=True)
                    elif output['type'] == 'result':
                        log_diagnostic(f"Received result from worker {process_id} with process ID {output['process_id']}")
                        if output['error']:
                            print(f"Execution error in process {output['process_id']}: {output['error']}", flush=True)
                        if output.get('warnings'):
                            for warning in output['warnings']:
                                print(f"[Process {output['process_id']}] Warning: {warning}", flush=True)
                        if output['vars']:
                            # Update local cache of namespace
                            self.shell.user_ns.update(output['vars'])
                            break
                except queue.Empty:
                    continue

        except KeyboardInterrupt:
            print("\nExecution interrupted by user. Sending interrupt to workers...")
            self._interrupt_workers(worker_n_procs, process_id=process_id)
            try:
                while True:
                    output = worker['output_queue'].get(timeout=1.0)
                    if output['type'] == 'result' and output['error'] == 'Execution interrupted':
                        break
            except queue.Empty:
                print(f"Process {process_id} did not respond to interrupt after 2 seconds.")
                return
            print(f"Process {process_id} interrupted successfully.")

    # def run_load_vars_distribute(
    #     self,
    #     var_list: List[str],
    #     process_id: int,
    #     debug: bool = False,
    #     cell: Optional[str] = None,
    #     to_pool_id: Optional[str] = None,
    #     to_process
    # ):      
    #     # Find the process pool with the smallest number of distributed processes
    #     if not self.workers:
    #         raise RuntimeError("No distributed workers available. Run %%distribute first to create worker processes.")
        
    #     worker_n_procs = min(self.workers.keys())
        
    #     # Validate process_id exists in the selected pool
    #     if process_id >= len(self.workers[worker_n_procs]):
    #         raise ValueError(f"Process ID {process_id} does not exist in the worker pool with {worker_n_procs} processes. "
    #                        f"Available process IDs: 0-{len(self.workers[worker_n_procs])-1}")
        
    #     worker = self.workers[worker_n_procs][process_id]

    #     if var_list:
    #         pass
    #     elif cell is None:
    #         raise ValueError("no variables specified to load. Use --vars to specify variable names or run with "
    #                          "`%%load_vars` so the current cell's code is available to extract variables.")
    #     else:
    #         var_list = find_undefined_variables(cell)

    #     diag_start = datetime.now()
    #     def log_diagnostic(message):
    #         if debug:
    #             elapsed_seconds = (datetime.now() - diag_start).total_seconds()
    #             elapsed_minutes = elapsed_seconds // 60
    #             elapsed_seconds %= 60
    #             relative_timestamp = f"{elapsed_minutes:.0f}:{elapsed_seconds:.5f}"
    #             print(f"[MAIN] at {relative_timestamp}: {message}\n")

    #     # query distributed process for variables
    #     worker['input_queue'].put({'type': 'load_vars', 'task': 'load_vars', 'vars': var_list, 'debug': debug})

    #     try:
    #         while True:
    #             try:
    #                 output = worker['output_queue'].get(timeout=0.01)

    #                 if output['type'] == 'stdout':
    #                     carriage_return = '\r'
    #                     print(f"[Process {output['process_id']}] {output['data'].strip(carriage_return)}", flush=True)
    #                 elif output['type'] == 'result':
    #                     log_diagnostic(f"Received result from worker {process_id} with process ID {output['process_id']}")
    #                     if output['error']:
    #                         print(f"Execution error in process {output['process_id']}: {output['error']}", flush=True)
    #                     if output.get('warnings'):
    #                         for warning in output['warnings']:
    #                             print(f"[Process {output['process_id']}] Warning: {warning}", flush=True)
    #                     if output['vars']:
    #                         # Update local cache of namespace
    #                         self.shell.user_ns.update(output['vars'])
    #                         break
    #             except queue.Empty:
    #                 continue

    #     except KeyboardInterrupt:
    #         print("\nExecution interrupted by user. Sending interrupt to workers...")
    #         self._interrupt_workers(worker_n_procs, process_id=process_id)
    #         try:
    #             while True:
    #                 output = worker['output_queue'].get(timeout=1.0)
    #                 if output['type'] == 'result' and output['error'] == 'Execution interrupted':
    #                     break
    #         except queue.Empty:
    #             print(f"Process {process_id} did not respond to interrupt after 2 seconds.")
    #             return
    #         print(f"Process {process_id} interrupted successfully.")

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
        if not code.startswith("%%distribute"):  # skip for distributed calls
            self.run_load_vars(
                var_list=[],      # infer variables to load from cell content
                process_id=0,
                debug=False,
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
