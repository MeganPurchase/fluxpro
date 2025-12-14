from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--update", action="store_true", default=False, help="Update golden (expected) output files"
    )


@pytest.fixture
def data_dir() -> Path:
    return Path("tests/test-data")
