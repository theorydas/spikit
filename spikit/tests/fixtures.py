from spikit.binary import BlackHole, Binary
from spikit.spike import StaticPowerLaw

import pytest

@pytest.fixture
def default_binary_spike_system():
    system = Binary(1e3, 1)
    spike = StaticPowerLaw(7/3, 1e16)
    
    return (system, spike)