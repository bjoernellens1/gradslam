from importlib import metadata

import gradslam


def test_version_matches_distribution():
    assert gradslam.__version__ == metadata.version("opengradslam")
