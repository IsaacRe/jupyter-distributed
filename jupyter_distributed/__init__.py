"""
Jupyter Distributed Extension

A Jupyter extension that provides the %distribute magic command for parallel execution of cells.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .magic import DistributedMagics

def load_ipython_extension(ipython):
    """Load the extension in IPython."""
    magics = DistributedMagics(ipython)
    ipython.register_magics(magics)
    print("Jupyter Distributed extension loaded. Use %distribute n to run cells in parallel.")

def unload_ipython_extension(ipython):
    """Unload the extension from IPython."""
    # Clean up any resources if needed
    pass

# Make the magic available for direct import
__all__ = ['DistributedMagics', 'load_ipython_extension', 'unload_ipython_extension']
