"""Regression test: import gradslam without segfault or open3d import."""

import sys
import subprocess


def test_import_without_open3d():
    """Ensure gradslam imports without requiring open3d at module level."""
    # Run in subprocess to isolate from current environment
    code = """
import sys
try:
    import open3d
    print("ERROR: open3d was imported at module level")
    sys.exit(1)
except ImportError:
    pass

import gradslam
print("OK: gradslam imported successfully without open3d")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Import failed: {result.stderr}"
    assert "OK" in result.stdout
