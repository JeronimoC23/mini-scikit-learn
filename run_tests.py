#!/usr/bin/env python3
"""
Simple test runner for mini-sklearn.

Usage:
    python run_tests.py           # Run all tests
    python run_tests.py ab        # Run only A/B tests
    python run_tests.py negative  # Run only negative/error tests
"""

import sys
import subprocess
import os

def run_command(cmd):
    """Run a command and return success status."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def main():
    # Set PYTHONPATH to include current directory
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()

    # Determine which tests to run
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
    else:
        test_type = "all"

    if test_type == "ab":
        # Run only A/B tests
        cmd = "PYTHONPATH=. python3 -m pytest -v -k 'ab'"
    elif test_type == "negative":
        # Run only negative/error tests
        cmd = "PYTHONPATH=. python3 -m pytest tests/test_negative_cases.py -v"
    elif test_type == "splits":
        cmd = "PYTHONPATH=. python3 -m pytest tests/test_split_stratified_ab.py -v"
    elif test_type == "minmax":
        cmd = "PYTHONPATH=. python3 -m pytest tests/test_minmax_ab.py -v"
    elif test_type == "forest":
        cmd = "PYTHONPATH=. python3 -m pytest tests/test_random_forest_ab.py -v"
    else:
        # Run all tests
        cmd = "PYTHONPATH=. python3 -m pytest tests/ -v"

    success = run_command(cmd)

    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()