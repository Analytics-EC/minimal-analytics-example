"""
Test runner script for running different test suites.
"""

import argparse
import subprocess
import sys


def run_tests(test_type: str = "all", verbose: bool = False) -> bool:
    """
    Run tests based on the specified type.

    Args:
        test_type: Type of tests to run ('unit', 'integration', 'all')
        verbose: Whether to run in verbose mode
    """
    cmd = ["python", "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    if test_type == "unit":
        cmd.append("tests/unit/")
    elif test_type == "integration":
        cmd.append("tests/integration/")
    elif test_type == "all":
        cmd.append("tests/")
    else:
        print(f"Unknown test type: {test_type}")
        return False

    print(f"Running {test_type} tests...")
    result = subprocess.run(cmd)
    return result.returncode == 0


def main() -> None:
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Run test suites")
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "all"],
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Run in verbose mode"
    )

    args = parser.parse_args()

    success = run_tests(args.type, args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
