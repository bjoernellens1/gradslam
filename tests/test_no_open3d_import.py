"""Regression test: import gradslam without segfault or open3d import."""

import subprocess
import sys


def test_import_without_open3d():
    """Ensure gradslam imports without pulling open3d into sys.modules."""
    code = """
import sys
import gradslam
if "open3d" in sys.modules:
    print("ERROR: open3d was imported during gradslam import")
    sys.exit(1)
print("OK: gradslam imported successfully without importing open3d")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Import failed: {result.stderr}"
    assert "OK" in result.stdout
