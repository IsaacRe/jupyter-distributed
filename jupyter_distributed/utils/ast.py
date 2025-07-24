"""
AST utilities for analyzing Python code variable usage.

This module provides functionality to identify undefined variables in Python code
by parsing the Abstract Syntax Tree (AST) and analyzing variable references
versus definitions.
"""

import ast
from typing import Set


# Python built-ins that should be excluded from undefined variable detection
PYTHON_BUILTINS = {
    # Built-in functions
    'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes',
    'callable', 'chr', 'classmethod', 'compile', 'complex', 'delattr',
    'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec', 'filter',
    'float', 'format', 'frozenset', 'getattr', 'globals', 'hasattr',
    'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance',
    'issubclass', 'iter', 'len', 'list', 'locals', 'map', 'max',
    'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord',
    'pow', 'print', 'property', 'range', 'repr', 'reversed', 'round',
    'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum',
    'super', 'tuple', 'type', 'vars', 'zip', '__import__',
    
    # Constants
    'True', 'False', 'None', 'Ellipsis', 'NotImplemented',
    
    # Common exceptions
    'Exception', 'ValueError', 'TypeError', 'KeyError', 'IndexError',
    'AttributeError', 'NameError', 'ImportError', 'ModuleNotFoundError',
    'RuntimeError', 'StopIteration', 'GeneratorExit', 'SystemExit',
    'KeyboardInterrupt', 'OSError', 'IOError', 'FileNotFoundError',
    
    # Other built-ins
    '__name__', '__doc__', '__package__', '__loader__', '__spec__',
    '__builtins__', '__file__', '__cached__'
}


class VariableReferenceCollector(ast.NodeVisitor):
    """Collects all variable names that are referenced (loaded) in the code."""
    
    def __init__(self):
        self.referenced = set()
    
    def visit_Name(self, node):
        """Visit Name nodes - these represent variable references."""
        # Only collect variables that are being loaded (not stored)
        if isinstance(node.ctx, ast.Load):
            self.referenced.add(node.id)
        self.generic_visit(node)
    
    def visit_Attribute(self, node):
        """Visit Attribute nodes - for obj.attr, we only care about 'obj'."""
        if isinstance(node.value, ast.Name) and isinstance(node.value.ctx, ast.Load):
            self.referenced.add(node.value.id)
        self.generic_visit(node)
    
    def visit_Subscript(self, node):
        """Visit Subscript nodes - for obj[key], we care about both 'obj' and 'key'."""
        # Handle the object being subscripted
        if isinstance(node.value, ast.Name) and isinstance(node.value.ctx, ast.Load):
            self.referenced.add(node.value.id)
        
        # Handle the slice/index - this might contain variable references
        self.generic_visit(node)


class VariableDefinitionCollector(ast.NodeVisitor):
    """Collects all variable names that are defined in the code."""
    
    def __init__(self):
        self.defined = set()
        self.scope_stack = [set()]  # Stack to handle nested scopes
    
    def _add_to_current_scope(self, name):
        """Add a variable to the current scope."""
        self.scope_stack[-1].add(name)
        self.defined.add(name)
    
    def visit_Name(self, node):
        """Visit Name nodes - collect variables that are being stored/assigned."""
        if isinstance(node.ctx, ast.Store):
            self._add_to_current_scope(node.id)
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        """Visit function definitions."""
        # Function name is defined in the current scope
        self._add_to_current_scope(node.name)
        
        # Create new scope for function body
        self.scope_stack.append(set())
        
        # Function parameters are defined within the function scope
        for arg in node.args.args:
            self._add_to_current_scope(arg.arg)
        
        # Handle keyword-only arguments
        for arg in node.args.kwonlyargs:
            self._add_to_current_scope(arg.arg)
        
        # Handle *args and **kwargs
        if node.args.vararg:
            self._add_to_current_scope(node.args.vararg.arg)
        if node.args.kwarg:
            self._add_to_current_scope(node.args.kwarg.arg)
        
        # Visit function body
        for stmt in node.body:
            self.visit(stmt)
        
        # Pop function scope
        self.scope_stack.pop()
    
    def visit_AsyncFunctionDef(self, node):
        """Visit async function definitions - same as regular functions."""
        self.visit_FunctionDef(node)
    
    def visit_ClassDef(self, node):
        """Visit class definitions."""
        # Class name is defined in the current scope
        self._add_to_current_scope(node.name)
        
        # Create new scope for class body
        self.scope_stack.append(set())
        
        # Visit class body
        for stmt in node.body:
            self.visit(stmt)
        
        # Pop class scope
        self.scope_stack.pop()
    
    def visit_Import(self, node):
        """Visit import statements."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name.split('.')[0]
            self._add_to_current_scope(name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Visit from-import statements."""
        for alias in node.names:
            if alias.name == '*':
                # Star imports are tricky - we can't know what they define
                continue
            name = alias.asname if alias.asname else alias.name
            self._add_to_current_scope(name)
        self.generic_visit(node)
    
    def visit_For(self, node):
        """Visit for loops - loop variables are defined."""
        self._visit_assignment_target(node.target)
        self.generic_visit(node)
    
    def visit_AsyncFor(self, node):
        """Visit async for loops - same as regular for loops."""
        self.visit_For(node)
    
    def visit_With(self, node):
        """Visit with statements - context manager variables are defined."""
        for item in node.items:
            if item.optional_vars:
                self._visit_assignment_target(item.optional_vars)
        self.generic_visit(node)
    
    def visit_AsyncWith(self, node):
        """Visit async with statements - same as regular with statements."""
        self.visit_With(node)
    
    def visit_ExceptHandler(self, node):
        """Visit exception handlers - exception variables are defined."""
        if node.name:
            self._add_to_current_scope(node.name)
        self.generic_visit(node)
    
    def visit_Lambda(self, node):
        """Visit lambda expressions - parameters are local to the lambda."""
        # Create new scope for lambda
        self.scope_stack.append(set())
        
        # Lambda parameters are defined within the lambda scope
        for arg in node.args.args:
            self._add_to_current_scope(arg.arg)
        
        # Handle keyword-only arguments
        for arg in node.args.kwonlyargs:
            self._add_to_current_scope(arg.arg)
        
        # Handle *args and **kwargs
        if node.args.vararg:
            self._add_to_current_scope(node.args.vararg.arg)
        if node.args.kwarg:
            self._add_to_current_scope(node.args.kwarg.arg)
        
        # Visit lambda body
        self.visit(node.body)
        
        # Pop lambda scope
        self.scope_stack.pop()
    
    def visit_ListComp(self, node):
        """Visit list comprehensions - they have their own scope."""
        self._visit_comprehension(node)
    
    def visit_SetComp(self, node):
        """Visit set comprehensions - they have their own scope."""
        self._visit_comprehension(node)
    
    def visit_DictComp(self, node):
        """Visit dict comprehensions - they have their own scope."""
        self._visit_comprehension(node)
    
    def visit_GeneratorExp(self, node):
        """Visit generator expressions - they have their own scope."""
        self._visit_comprehension(node)
    
    def _visit_comprehension(self, node):
        """Handle comprehensions which have isolated scopes."""
        # Create new scope for comprehension
        self.scope_stack.append(set())
        
        # Visit generators (for clauses)
        for generator in node.generators:
            # The target variables are defined in the comprehension scope
            self._visit_assignment_target(generator.target)
            # Visit the iterator and conditions
            self.visit(generator.iter)
            for if_clause in generator.ifs:
                self.visit(if_clause)
        
        # Visit the element expression (or key/value for dict comp)
        if hasattr(node, 'elt'):  # ListComp, SetComp, GeneratorExp
            self.visit(node.elt)
        elif hasattr(node, 'key'):  # DictComp
            self.visit(node.key)
            self.visit(node.value)
        
        # Pop comprehension scope
        self.scope_stack.pop()
    
    def _visit_assignment_target(self, target):
        """Visit assignment targets (can be complex patterns)."""
        if isinstance(target, ast.Name):
            self._add_to_current_scope(target.id)
        elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
            for elt in target.elts:
                self._visit_assignment_target(elt)
        elif isinstance(target, ast.Starred):
            self._visit_assignment_target(target.value)
        # For other types (Attribute, Subscript), we don't define new variables


def find_undefined_variables(code: str) -> Set[str]:
    """
    Find variables that are referenced but not defined in the given code.
    
    Args:
        code: Python code as a string
        
    Returns:
        Set of variable names that are used but not defined in the code
        
    Raises:
        SyntaxError: If the code cannot be parsed
    """
    if not code or not code.strip():
        return set()
    
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # If code has syntax errors, return empty set
        return set()
    
    # Collect referenced and defined variables
    ref_collector = VariableReferenceCollector()
    def_collector = VariableDefinitionCollector()
    
    ref_collector.visit(tree)
    def_collector.visit(tree)
    
    # Find undefined variables (referenced but not defined, excluding built-ins)
    undefined = ref_collector.referenced - def_collector.defined - PYTHON_BUILTINS
    
    return undefined
