#!/usr/bin/env python3
"""
Test script to verify the AST functionality for finding undefined variables.
"""

from jupyter_distributed.utils.ast import find_undefined_variables

def test_basic_functionality():
    """Test basic variable detection."""
    print("Testing basic functionality...")
    
    # Test 1: Simple undefined variable
    code1 = "print(x + y)"
    result1 = find_undefined_variables(code1)
    print(f"Code: {code1}")
    print(f"Undefined variables: {result1}")
    assert result1 == {'x', 'y'}, f"Expected {{'x', 'y'}}, got {result1}"
    
    # Test 2: Variable defined in same code
    code2 = """
x = 5
y = 10
print(x + y)
"""
    result2 = find_undefined_variables(code2)
    print(f"Code: {repr(code2)}")
    print(f"Undefined variables: {result2}")
    assert result2 == set(), f"Expected empty set, got {result2}"
    
    # Test 3: Mix of defined and undefined
    code3 = """
x = 5
print(x + y + z)
"""
    result3 = find_undefined_variables(code3)
    print(f"Code: {repr(code3)}")
    print(f"Undefined variables: {result3}")
    assert result3 == {'y', 'z'}, f"Expected {{'y', 'z'}}, got {result3}"
    
    print("✓ Basic functionality tests passed!\n")

def test_advanced_cases():
    """Test more advanced cases."""
    print("Testing advanced cases...")
    
    # Test 4: Function definitions
    code4 = """
def my_func(a, b):
    return a + b + external_var

result = my_func(1, 2)
"""
    result4 = find_undefined_variables(code4)
    print(f"Code: {repr(code4)}")
    print(f"Undefined variables: {result4}")
    assert result4 == {'external_var'}, f"Expected {{'external_var'}}, got {result4}"
    
    # Test 5: List comprehensions
    code5 = """
numbers = [1, 2, 3]
squared = [x**2 for x in numbers]
filtered = [item for item in data if item > threshold]
"""
    result5 = find_undefined_variables(code5)
    print(f"Code: {repr(code5)}")
    print(f"Undefined variables: {result5}")
    assert result5 == {'data', 'threshold'}, f"Expected {{'data', 'threshold'}}, got {result5}"
    
    # Test 6: Imports
    code6 = """
import numpy as np
from math import sqrt
result = np.array([1, 2, 3]) + unknown_var
"""
    result6 = find_undefined_variables(code6)
    print(f"Code: {repr(code6)}")
    print(f"Undefined variables: {result6}")
    assert result6 == {'unknown_var'}, f"Expected {{'unknown_var'}}, got {result6}"
    
    print("✓ Advanced cases tests passed!\n")

def test_edge_cases():
    """Test edge cases."""
    print("Testing edge cases...")
    
    # Test 7: Empty code
    code7 = ""
    result7 = find_undefined_variables(code7)
    print(f"Code: {repr(code7)}")
    print(f"Undefined variables: {result7}")
    assert result7 == set(), f"Expected empty set, got {result7}"
    
    # Test 8: Only comments
    code8 = "# This is just a comment"
    result8 = find_undefined_variables(code8)
    print(f"Code: {repr(code8)}")
    print(f"Undefined variables: {result8}")
    assert result8 == set(), f"Expected empty set, got {result8}"
    
    # Test 9: Built-ins should be excluded
    code9 = "result = len(my_list) + max(my_data)"
    result9 = find_undefined_variables(code9)
    print(f"Code: {repr(code9)}")
    print(f"Undefined variables: {result9}")
    assert result9 == {'my_list', 'my_data'}, f"Expected {{'my_list', 'my_data'}}, got {result9}"
    
    print("✓ Edge cases tests passed!\n")

if __name__ == "__main__":
    print("Testing AST functionality for finding undefined variables...\n")
    
    try:
        test_basic_functionality()
        test_advanced_cases()
        test_edge_cases()
        print("🎉 All tests passed! The AST functionality is working correctly.")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
