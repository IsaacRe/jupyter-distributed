#!/usr/bin/env python3
"""
Simple test script to verify the extension works without IPython testing dependencies.
"""

from jupyter_distributed.magic import DistributedMagics
from IPython.core.magic_arguments import parse_argstring


def test_basic_functionality():
    """Test basic functionality without full IPython setup."""
    print("Testing basic extension functionality...")
    
    # Test magic class instantiation
    try:
        magic = DistributedMagics()
        print("✓ DistributedMagics class instantiated successfully")
    except Exception as e:
        print(f"✗ Failed to instantiate DistributedMagics: {e}")
        return False
    
    # Test argument parsing
    try:
        args = parse_argstring(DistributedMagics.distribute, "4")
        print(f"✓ Argument parsing works: n_processes={args.n_processes}")
    except Exception as e:
        print(f"✗ Argument parsing failed: {e}")
        return False
    
    # Test argument parsing with flags
    try:
        args = parse_argstring(DistributedMagics.distribute, "4 --sync --timeout 10")
        print(f"✓ Advanced argument parsing works: n_processes={args.n_processes}, sync={args.sync}, timeout={args.timeout}")
    except Exception as e:
        print(f"✗ Advanced argument parsing failed: {e}")
        return False
    
    # Test variable extraction
    try:
        code = "x = 5\ny = 10\nresult = x + y"
        assignments = magic._extract_assignments(code)
        expected = ['x', 'y', 'result']
        if set(assignments) == set(expected):
            print(f"✓ Variable extraction works: {assignments}")
        else:
            print(f"✗ Variable extraction failed: expected {expected}, got {assignments}")
            return False
    except Exception as e:
        print(f"✗ Variable extraction failed: {e}")
        return False
    
    print("✓ All basic functionality tests passed!")
    return True


if __name__ == '__main__':
    test_basic_functionality()
