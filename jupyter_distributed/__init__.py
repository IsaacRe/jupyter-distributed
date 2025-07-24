"""
Jupyter Distributed Extension

A Jupyter extension that provides the %distribute magic command for parallel execution of cells.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .magic import DistributedMagics

_magic_instance = None


def load_ipython_extension(ipython):
    """Load the extension in IPython."""
    global _magic_instance
    if _magic_instance is None:
        _magic_instance = DistributedMagics(ipython)
        ipython.register_magics(_magic_instance)
        ipython.events.register('pre_run_cell', _magic_instance.pre_cell_hook)
        print("Jupyter Distributed extension with persistent processes loaded.")
    else:
        print("Jupyter Distributed extension already loaded.")

def unload_ipython_extension(ipython):
    """Unload the extension from IPython and clean up resources."""
    global _magic_instance
    if _magic_instance:
        # Clean up worker processes
        _magic_instance._cleanup_workers()
        ipython.events.unregister('pre_run_cell', _magic_instance.pre_cell_hook)
        _magic_instance = None
        print("Jupyter Distributed extension unloaded and processes cleaned up.")

# Make the magic available for direct import
__all__ = ['DistributedMagics', 'load_ipython_extension', 'unload_ipython_extension']
