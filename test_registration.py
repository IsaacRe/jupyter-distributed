#!/usr/bin/env python3
"""
Test script to verify magic registration works correctly.
"""

from IPython.testing import tools as tt
from IPython.core.magic import register_line_cell_magic
from jupyter_distributed import load_ipython_extension
from jupyter_distributed.magic import DistributedMagics


def test_magic_registration():
    """Test that the magic can be registered and called."""
    print("Testing magic registration...")
    
    # Create a mock IPython instance
    ip = tt.get_ipython()
    
    # Load the extension
    try:
        load_ipython_extension(ip)
        print("✓ Extension loaded successfully")
    except Exception as e:
        print(f"✗ Extension loading failed: {e}")
        return False
    
    # Check if the magic is registered
    if 'distribute' in ip.magics_manager.magics['line']:
        print("✓ Line magic 'distribute' registered successfully")
    else:
        print("✗ Line magic 'distribute' not found")
        return False
        
    if 'distribute' in ip.magics_manager.magics['cell']:
        print("✓ Cell magic 'distribute' registered successfully")
    else:
        print("✗ Cell magic 'distribute' not found")
        return False
    
    # Test basic argument parsing
    try:
        magic_instance = DistributedMagics(ip)
        from IPython.core.magic_arguments import parse_argstring
        args = parse_argstring(DistributedMagics.distribute, "4")
        print(f"✓ Argument parsing works: n_processes={args.n_processes}")
    except Exception as e:
        print(f"✗ Argument parsing failed: {e}")
        return False
    
    print("✓ All registration tests passed!")
    return True


if __name__ == '__main__':
    test_magic_registration()
