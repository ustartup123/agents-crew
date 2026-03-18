"""
tools/code_exec_tools.py — E2B sandbox tools for the Dev and QA agents.

The sandbox is a secure cloud microVM where code can be written, executed,
and tested. Results (stdout, stderr, errors) are returned to the agent.
"""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool
from e2b_code_interpreter import Sandbox

from config.settings import e2b_cfg

logger = logging.getLogger(__name__)

# Module-level sandbox instance — reused within a single agent node execution.
_active_sandbox: Sandbox | None = None


def get_sandbox() -> Sandbox:
    """Get or create the active E2B sandbox for this session."""
    global _active_sandbox
    if _active_sandbox is None:
        _active_sandbox = Sandbox(api_key=e2b_cfg.api_key, timeout=300)
    return _active_sandbox


def close_sandbox():
    """Close and destroy the active sandbox."""
    global _active_sandbox
    if _active_sandbox is not None:
        try:
            _active_sandbox.kill()
        except Exception as e:
            logger.warning(f"Error closing sandbox: {e}")
        _active_sandbox = None


@tool
def run_python_code(code: str) -> str:
    """Execute Python code in a secure cloud sandbox and return the output.
    Args: code (Python code to execute).
    Returns: JSON with stdout, stderr, error (if any)."""
    sandbox = get_sandbox()
    execution = sandbox.run_code(code)
    return json.dumps({
        "stdout": execution.logs.stdout,
        "stderr": execution.logs.stderr,
        "error": str(execution.error) if execution.error else None,
    })


@tool
def run_shell_command(command: str) -> str:
    """Run a shell command in the sandbox (e.g., install packages, run tests, git ops).
    Args: command (shell command string, e.g. 'pip install requests' or 'pytest tests/').
    Returns: JSON with stdout, stderr, exit_code."""
    sandbox = get_sandbox()
    result = sandbox.commands.run(command, timeout=120)
    return json.dumps({
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.exit_code,
    })


@tool
def write_sandbox_file(file_path: str, content: str) -> str:
    """Write a file to the sandbox filesystem.
    Args: file_path (e.g. 'src/main.py'), content (file content as string).
    Returns: confirmation message."""
    sandbox = get_sandbox()
    sandbox.files.write(file_path, content)
    return f"File written: {file_path}"


@tool
def read_sandbox_file(file_path: str) -> str:
    """Read a file from the sandbox filesystem.
    Args: file_path.
    Returns: file content as string."""
    sandbox = get_sandbox()
    return sandbox.files.read(file_path)


@tool
def install_package(package_name: str) -> str:
    """Install a Python package in the sandbox.
    Args: package_name (e.g. 'requests' or 'fastapi uvicorn').
    Returns: installation result."""
    sandbox = get_sandbox()
    result = sandbox.commands.run(f"pip install {package_name} -q", timeout=60)
    return json.dumps({
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.exit_code,
    })


@tool
def run_tests(test_command: str = "pytest") -> str:
    """Run the test suite in the sandbox.
    Args: test_command (default 'pytest', can be 'pytest tests/ -v' or 'python -m pytest').
    Returns: JSON with test output, pass/fail status."""
    sandbox = get_sandbox()
    result = sandbox.commands.run(test_command, timeout=120)
    passed = result.exit_code == 0
    return json.dumps({
        "passed": passed,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.exit_code,
    })


CODE_EXEC_TOOLS = [run_python_code, run_shell_command, write_sandbox_file,
                   read_sandbox_file, install_package, run_tests]
