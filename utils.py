#!/usr/bin/env python3
"""
utils.py
~~~~~~~~
Shared utility functions for the MoJ Dashboard CLI.
"""

import subprocess
import sys
from typing import List


def backend_run(command: List[str], env: dict = None) -> None:
    """Run a command in the backend directory and exit with its return code."""
    result = subprocess.run(command, cwd="backend", env=env)
    sys.exit(result.returncode)


def frontend_run(command: List[str]) -> None:
    """Run a command in the frontend directory and exit with its return code."""
    result = subprocess.run(command, cwd="frontend")
    sys.exit(result.returncode)


def ensure_backend_deps() -> None:
    """Install backend dependencies via Poetry."""
    subprocess.run(["poetry", "install", "--extras", "dev"], cwd="backend")


def ensure_frontend_deps() -> None:
    """Install frontend dependencies via npm if not already installed."""
    import os
    if not os.path.exists("frontend/node_modules"):
        subprocess.run(["npm", "install"], cwd="frontend")


def get_commit() -> str:
    """Get the current git commit hash."""
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip()


def get_aws_env() -> dict:
    """Get AWS credentials as environment variables."""
    import os
    aws_creds = subprocess.run(
        ["aws", "configure", "export-credentials", "--profile", os.environ.get("AWS_PROFILE", ""), "--format", "env"],
        capture_output=True, text=True
    )
    return {**os.environ, **dict(line.split("=", 1) for line in aws_creds.stdout.splitlines() if "=" in line)}
