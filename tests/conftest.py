def pytest_addoption(parser):
    parser.addoption(
        "--update", action="store_true", default=False, help="Update golden (expected) output files"
    )
