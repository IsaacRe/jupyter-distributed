#!/usr/bin/env python3
"""
Test script to verify stdout capture functionality in the distribute magic.
"""

import sys
import os

# Add the current directory to Python path so we can import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jupyter_distributed.magic import DistributedMagics

def test_stdout_capture():
    """Test that stdout from processes is captured and displayed."""
    print("Testing stdout capture functionality...")
    
    # Create a magic instance
    magic = DistributedMagics()
    
    # Test code that produces output
    test_code = '''
import os
print(f"Hello from process {__process_id__}!")
print(f"Process ID: {os.getpid()}")
result = __process_id__ * 2
print(f"Result: {result}")
'''
    
    print("\n" + "="*50)
    print("Running distribute magic with stdout capture...")
    print("="*50)
    
    # Execute the distribute magic
    magic.distribute("3", test_code)
    
    print("\n" + "="*50)
    print("Test completed!")
    print("="*50)

if __name__ == "__main__":
    test_stdout_capture()
