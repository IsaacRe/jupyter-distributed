#!/usr/bin/env python3
"""
Demo script for jupyter-distributed extension.
This script demonstrates the extension functionality outside of Jupyter.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jupyter_distributed.magic import DistributedMagics


def demo_basic_functionality():
    """Demonstrate basic functionality of the distributed magic."""
    print("Jupyter Distributed Extension Demo")
    print("=" * 40)
    
    # Create magic instance
    magic = DistributedMagics()
    
    # Test variable extraction
    print("\n1. Testing variable extraction:")
    code = """
x = 5
y = 10
result = x + y
"""
    assignments = magic._extract_assignments(code)
    print(f"Code:\n{code}")
    print(f"Extracted assignments: {assignments}")
    
    # Test process execution
    print("\n2. Testing process execution:")
    test_code = """
import os
process_result = __process_id__ * 2
message = f"Hello from process {__process_id__}! PID: {os.getpid()}"
"""
    
    # Test with a simple namespace
    namespace = {}
    process_id = 1
    
    try:
        pid, result_vars, error = magic._execute_in_process((test_code, namespace, process_id))
        print(f"Process ID: {pid}")
        print(f"Error: {error}")
        print(f"Result variables: {result_vars}")
    except Exception as e:
        print(f"Error during execution: {e}")
    
    print("\n3. Testing synchronization:")
    # Set up test namespaces
    magic.process_namespaces[2] = [
        {'x': 100, 'shared_var': 'from_process_0'},
        {'x': 200, 'other_var': 'from_process_1'}
    ]
    
    print("Before sync:")
    print(f"Process 0: {magic.process_namespaces[2][0]}")
    print(f"Process 1: {magic.process_namespaces[2][1]}")
    
    # Sync variables
    magic._sync_variables(2, ['shared_var'])
    
    print("After sync:")
    print(f"Process 0: {magic.process_namespaces[2][0]}")
    print(f"Process 1: {magic.process_namespaces[2][1]}")
    
    print("\n✓ Demo completed successfully!")


if __name__ == '__main__':
    demo_basic_functionality()
