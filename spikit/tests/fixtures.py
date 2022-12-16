from spikit.binary import BlackHole, Binary
from spikit.spike import StaticPowerLaw

import pytest

@pytest.fixture
def default_binary_spike_system():
    m1 = BlackHole(1e3); m2 = BlackHole(1)
    system = Binary(m1, m2, 100)
    spike = StaticPowerLaw(m1, 7/3, 1e16)
    
    return (system, spike)