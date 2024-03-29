from spikit.binary import Binary
from spikit.spike import StaticPowerLaw, PowerLaw

import pytest

@pytest.fixture
def default_binary() -> Binary:
    binary = Binary(m1 = 1e3, m2 = 1)
    
    return binary

@pytest.fixture
def default_spike() -> StaticPowerLaw:
    spike = StaticPowerLaw(m1 = 1e3, m2 = 1, gammasp = 7/3, rho6 = 1e16)
    
    return spike

@pytest.fixture
def default_dynaspike() -> PowerLaw:
    spike = PowerLaw(m1 = 1e3, m2 = 1, gammasp = 7/3, rho6 = 1e16)
    
    return spike