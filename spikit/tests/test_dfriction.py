from spikit.forces import DynamicalFriction, DynamicalFrictionIso
from spikit.binary import Binary
from spikit.spike import Spike

from spikit.tests.fixtures import default_binary, default_spike

def test_iso_accretion_force(default_binary: Binary, default_spike: Spike):
    df = DynamicalFrictionIso(default_binary, default_spike)
    df0 = DynamicalFriction(default_binary, default_spike)
    
    assert df.F(1, 1) == df0.F(1, 1) *df.xi_DF(1, 1) # [N]