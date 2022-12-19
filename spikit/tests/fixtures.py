from spikit.binary import Binary
from spikit.spike import StaticPowerLaw, Spike

import pytest

@pytest.fixture
def default_binary() -> Binary:
    system = Binary(1e3, 1)
    
    return system

@pytest.fixture
def default_spike() -> Spike:
    spike = StaticPowerLaw(7/3, 1e16)
    
    return spike