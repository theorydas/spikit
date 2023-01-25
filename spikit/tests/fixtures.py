from spikit.binary import Binary
from spikit.spike import StaticPowerLaw, Spike

import pytest

@pytest.fixture
def default_binary() -> Binary:
    binary = Binary(m1 = 1e3, m2 = 1)
    
    return binary

@pytest.fixture
def default_spike() -> Spike:
    binary = Binary(m1 = 1e3, m2 = 1)
    spike = StaticPowerLaw(binary = binary, gammasp = 7/3, rho6 = 1e16)
    
    return spike