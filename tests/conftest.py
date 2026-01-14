from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--update", action="store_true", default=False, help="Update golden (expected) output files"
    )


@pytest.fixture
def update_expected(request) -> bool:
    return request.config.getoption("--update")


@pytest.fixture
def data_dir() -> Path:
    return Path(__file__).parent.resolve() / "test-data"
