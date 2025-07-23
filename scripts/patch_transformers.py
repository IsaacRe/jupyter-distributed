#!/usr/bin/env python3
"""
Script to comment out specific lines in transformers library files.
This script finds the transformers installation location and comments out
the specified code blocks in modeling_utils.py and tensor_parallel.py.
"""

import os
import sys
import re
import importlib.util


def find_transformers_location():
    """Find the installation location of the transformers library."""
    try:
        import transformers
        transformers_path = os.path.dirname(transformers.__file__)
        return transformers_path
    except ImportError:
        print("Error: transformers library not found. Please install it first.")
        sys.exit(1)


def comment_out_modeling_utils_lines(file_path):
    """Comment out the specified lines in modeling_utils.py."""
    print(f"Processing {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return False
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

    # Look for the specific code block to comment out
    target_lines = [
        "        # if tp_plan is not None and tp_plan != \"auto\":",
        "#     # TODO: we can relax this check when we support taking tp_plan from a json file, for example.",
        "#     raise ValueError(f\"tp_plan supports 'auto' only for now but got {tp_plan}.\")"
    ]
    
    # Check if already commented
    already_commented = all(line in content for line in target_lines)
    if already_commented:
        print(f"Patch was already applied to {file_path}.")
        return True
    
    # Look for the uncommented version
    uncommented_block = '''        if tp_plan is not None and tp_plan != "auto":
            # TODO: we can relax this check when we support taking tp_plan from a json file, for example.
            raise ValueError(f"tp_plan supports 'auto' only for now but got {tp_plan}.")'''
    
    commented_block = '''        # if tp_plan is not None and tp_plan != "auto":
        #     # TODO: we can relax this check when we support taking tp_plan from a json file, for example.
        #     raise ValueError(f"tp_plan supports 'auto' only for now but got {tp_plan}.")'''
    
    if uncommented_block in content:
        new_content = content.replace(uncommented_block, commented_block)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Successfully patched {file_path}")
            return True
        except Exception as e:
            print(f"Error writing to {file_path}: {e}")
            return False
    else:
        print(f"Target code block not found in {file_path}. The code may have already been modified or the structure has changed.")
        return False


def comment_out_tensor_parallel_lines(file_path):
    """Comment out the specified lines in tensor_parallel.py."""
    print(f"Processing {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return False
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

    # Check if already commented (the lines are already commented in the current file)
    already_commented_block = '''    # # Silence output for non-primary ranks
    # if index is not None and index > 0:
    #     import sys

    #     sys.stdout = open(os.devnull, "w")
    #     sys.stderr = open(os.devnull, "w")'''
    
    if already_commented_block in content:
        print(f"Patch was already applied to {file_path}.")
        return True
    
    # Look for the uncommented version (if it exists)
    uncommented_block = '''    # Silence output for non-primary ranks
    if index is not None and index > 0:
        import sys

        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")'''
    
    commented_block = '''    # # Silence output for non-primary ranks
    # if index is not None and index > 0:
    #     import sys

    #     sys.stdout = open(os.devnull, "w")
    #     sys.stderr = open(os.devnull, "w")'''
    
    if uncommented_block in content:
        new_content = content.replace(uncommented_block, commented_block)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Successfully patched {file_path}")
            return True
        except Exception as e:
            print(f"Error writing to {file_path}: {e}")
            return False
    else:
        print(f"Target code block not found in {file_path}. The code may have already been modified or the structure has changed.")
        return False


def main():
    """Main function to execute the script."""
    print("Finding transformers installation location...")
    transformers_path = find_transformers_location()
    print(f"Found transformers at: {transformers_path}")
    
    # Define the file paths
    modeling_utils_path = os.path.join(transformers_path, 'modeling_utils.py')
    tensor_parallel_path = os.path.join(transformers_path, 'integrations', 'tensor_parallel.py')
    
    success_count = 0
    
    # Process modeling_utils.py
    if comment_out_modeling_utils_lines(modeling_utils_path):
        success_count += 1
    else:
        print(f"Failed to patch {modeling_utils_path}.")
    
    # Process tensor_parallel.py
    if comment_out_tensor_parallel_lines(tensor_parallel_path):
        success_count += 1
    else:
        print(f"Failed to patch {tensor_parallel_path}.")
    
    print(f"\nCompleted processing. Successfully patched {success_count}/2 files.")
    
    if success_count == 2:
        print("Patch applied successfully.")
    elif success_count == 1:
        print("Failed to apply patch! Check logs to see which file(s) failed.")
    else:
        print("Failed to apply patch! Check logs to see which file(s) failed.")


if __name__ == "__main__":
    main()
