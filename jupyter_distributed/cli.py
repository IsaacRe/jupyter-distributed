"""
Command-line interface for jupyter-distributed extension.
"""

import argparse
import sys
from IPython import get_ipython


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Jupyter Distributed Extension CLI')
    parser.add_argument('--install', action='store_true', 
                       help='Install the extension in the current IPython session')
    parser.add_argument('--version', action='store_true',
                       help='Show version information')
    
    args = parser.parse_args()
    
    if args.version:
        try:
            from ._version import version
            print(f"jupyter-distributed version {version}")
        except ImportError:
            print("jupyter-distributed version unknown")
        return
    
    if args.install:
        try:
            ipython = get_ipython()
            if ipython is None:
                print("Error: Not running in an IPython environment")
                sys.exit(1)
            
            from . import load_ipython_extension
            load_ipython_extension(ipython)
            print("Extension installed successfully!")
        except Exception as e:
            print(f"Error installing extension: {e}")
            sys.exit(1)
        return
    
    parser.print_help()


if __name__ == '__main__':
    main()
