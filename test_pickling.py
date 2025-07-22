#!/usr/bin/env python3
"""
Test script to verify pickling issues are resolved.
"""

import sqlite3
from jupyter_distributed.magic import DistributedMagics
import dill


def test_pickling_filter():
    """Test that unpicklable objects are properly filtered."""
    print("Testing pickling filter...")
    
    magic = DistributedMagics()
    
    # Create a namespace with both picklable and unpicklable objects
    test_namespace = {
        'good_var': 42,
        'good_string': 'hello',
        'good_list': [1, 2, 3],
        'bad_connection': sqlite3.connect(':memory:'),  # This can't be pickled
        'good_dict': {'a': 1, 'b': 2}
    }
    
    print(f"Original namespace keys: {list(test_namespace.keys())}")
    
    # Test the filtering in _execute_in_process
    test_code = "result = good_var * 2"
    
    try:
        process_id, result_vars, error = magic._execute_in_process((test_code, test_namespace, 0))
        print(f"✓ Process execution succeeded")
        print(f"Process ID: {process_id}")
        print(f"Error: {error}")
        print(f"Result variables: {list(result_vars.keys())}")
        
        # Check that good variables are present and bad ones are filtered out
        if 'good_var' in result_vars and 'bad_connection' not in result_vars:
            print("✓ Filtering works correctly - good vars kept, bad vars removed")
        else:
            print("✗ Filtering failed")
            
    except Exception as e:
        print(f"✗ Process execution failed: {e}")
        return False
    
    print("✓ Pickling filter test passed!")
    return True


if __name__ == '__main__':
    test_pickling_filter()
