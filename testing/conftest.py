import pytest

def pytest_addoption(parser):
    parser.addoption("--endpoint_url", action="store", default="")

@pytest.fixture
def endpoint_url(pytestconfig):
    return pytestconfig.getoption("endpoint_url")