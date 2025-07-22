"""
Tests for the jupyter-distributed magic commands.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch
from IPython.testing import tools as tt
from IPython.core.magic import register_line_cell_magic

# Add the parent directory to the path so we can import our module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jupyter_distributed.magic import DistributedMagics


class TestDistributedMagics:
    """Test cases for DistributedMagics class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.magic = DistributedMagics()
    
    def test_extract_assignments_simple(self):
        """Test extraction of simple variable assignments."""
        code = """
x = 5
y = 10
z = x + y
"""
        assignments = self.magic._extract_assignments(code)
        assert set(assignments) == {'x', 'y', 'z'}
    
    def test_extract_assignments_tuple_unpacking(self):
        """Test extraction of tuple unpacking assignments."""
        code = """
a, b = 1, 2
c, d, e = some_function()
"""
        assignments = self.magic._extract_assignments(code)
        assert set(assignments) == {'a', 'b', 'c', 'd', 'e'}
    
    def test_extract_assignments_augmented(self):
        """Test extraction of augmented assignments."""
        code = """
x += 5
y *= 2
z //= 3
"""
        assignments = self.magic._extract_assignments(code)
        assert set(assignments) == {'x', 'y', 'z'}
    
    def test_extract_assignments_annotated(self):
        """Test extraction of annotated assignments."""
        code = """
x: int = 5
y: str = "hello"
z: float
"""
        assignments = self.magic._extract_assignments(code)
        assert set(assignments) == {'x', 'y', 'z'}
    
    def test_extract_assignments_invalid_syntax(self):
        """Test that invalid syntax returns empty list."""
        code = "x = 5 +"  # Invalid syntax
        assignments = self.magic._extract_assignments(code)
        assert assignments == []
    
    def test_get_or_create_pool(self):
        """Test process pool creation and retrieval."""
        # First call should create the pool
        pool1, namespaces1 = self.magic._get_or_create_pool(4)
        assert 4 in self.magic.process_pools
        assert 4 in self.magic.process_namespaces
        assert len(namespaces1) == 4
        
        # Second call should return the same pool
        pool2, namespaces2 = self.magic._get_or_create_pool(4)
        assert pool1 is pool2
        assert namespaces1 is namespaces2
    
    def test_sync_variables(self):
        """Test variable synchronization across processes."""
        # Set up namespaces
        self.magic.process_namespaces[2] = [
            {'x': 10, 'y': 'hello'},
            {'x': 5, 'z': 'world'}
        ]
        
        # Sync variables
        self.magic._sync_variables(2, ['x', 'y'])
        
        # Check that process 1 now has the values from process 0
        assert self.magic.process_namespaces[2][1]['x'] == 10
        assert self.magic.process_namespaces[2][1]['y'] == 'hello'
        # z should remain unchanged
        assert self.magic.process_namespaces[2][1]['z'] == 'world'
    
    def test_execute_in_process(self):
        """Test code execution in a separate process context."""
        code = """
result = __process_id__ * 2
message = f"Process {__process_id__}"
"""
        namespace = {'existing_var': 42}
        process_id = 1
        
        # Create a mock queue for the new signature
        from unittest.mock import Mock
        mock_queue = Mock()
        
        pid, result_vars, error = self.magic._execute_in_process((code, namespace, process_id, mock_queue))
        
        assert pid == 1
        assert error is None
        assert result_vars['result'] == 2
        assert result_vars['message'] == "Process 1"
        assert result_vars['__process_id__'] == 1
        assert result_vars['existing_var'] == 42
    
    def test_execute_in_process_with_error(self):
        """Test error handling in process execution."""
        code = "x = 1 / 0"  # Division by zero
        namespace = {}
        process_id = 0
        
        # Create a mock queue for the new signature
        from unittest.mock import Mock
        mock_queue = Mock()
        
        pid, result_vars, error = self.magic._execute_in_process((code, namespace, process_id, mock_queue))
        
        assert pid == 0
        assert result_vars == {}
        assert "division by zero" in error.lower()
    
    def test_execute_in_process_with_stdout(self):
        """Test stdout streaming in process execution."""
        code = """
print(f"Hello from process {__process_id__}")
print("Multiple lines")
print("of output")
result = __process_id__ * 3
"""
        namespace = {}
        process_id = 2
        
        # Create a mock queue for the new signature
        from unittest.mock import Mock
        mock_queue = Mock()
        
        pid, result_vars, error = self.magic._execute_in_process((code, namespace, process_id, mock_queue))
        
        assert pid == 2
        assert error is None
        assert result_vars['result'] == 6
        assert result_vars['__process_id__'] == 2
        
        # Check that output was sent to the queue with proper prefixes
        expected_calls = [
            "[Process 2] Hello from process 2",
            "[Process 2] Multiple lines", 
            "[Process 2] of output"
        ]
        
        # Verify that put was called with the expected prefixed output
        assert mock_queue.put.call_count == 3
        actual_calls = [call[0][0] for call in mock_queue.put.call_args_list]
        assert actual_calls == expected_calls


class TestDistributedMagicsIntegration:
    """Integration tests for the magic commands."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.magic = DistributedMagics()
    
    @patch('multiprocess.Manager')
    @patch('multiprocess.Pool')
    def test_distribute_basic(self, mock_pool_class, mock_manager_class):
        """Test basic distribute command functionality."""
        # Mock the pool and its map method
        mock_pool = Mock()
        mock_pool_class.return_value = mock_pool
        mock_pool.map.return_value = [
            (0, {'result': 0}, None),  # Updated to new signature: (process_id, result_vars, error)
            (1, {'result': 1}, None)   # Updated to new signature: (process_id, result_vars, error)
        ]
        
        # Mock the manager and queue
        mock_manager = Mock()
        mock_queue = Mock()
        mock_manager.Queue.return_value = mock_queue
        mock_manager_class.return_value = mock_manager
        
        # Test the distribute command
        line = "2"
        cell = "result = __process_id__"
        
        # Capture output
        with patch('builtins.print') as mock_print:
            self.magic.distribute(line, cell)
        
        # Verify pool was created and used
        mock_pool_class.assert_called_once_with(2)
        mock_pool.map.assert_called_once()
        
        # Verify output messages
        print_calls = [str(call[0][0]) for call in mock_print.call_args_list]
        assert any("Distributing execution across 2 processes" in call for call in print_calls)
        assert any("Successfully executed in all 2 processes" in call for call in print_calls)
    
    def test_distribute_invalid_processes(self):
        """Test distribute command with invalid process count."""
        line = "0"  # Invalid: must be positive
        cell = "x = 1"
        
        with patch('builtins.print') as mock_print:
            self.magic.distribute(line, cell)
        
        mock_print.assert_called_with("Error: Number of processes must be positive")
    
    def test_distribute_empty_code(self):
        """Test distribute command with empty code."""
        line = "2"
        cell = ""  # Empty cell
        
        with patch('builtins.print') as mock_print:
            self.magic.distribute(line, cell)
        
        mock_print.assert_called_with("Error: No code to execute")


if __name__ == '__main__':
    pytest.main([__file__])
