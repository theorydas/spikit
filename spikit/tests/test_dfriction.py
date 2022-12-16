from spikit.forces import DynamicalFriction, DynamicalFrictionIso

from spikit.tests.fixtures import default_binary_spike_system

def test_iso_accretion_force(default_binary_spike_system):
    df = DynamicalFrictionIso(*default_binary_spike_system)
    df0 = DynamicalFriction(*default_binary_spike_system)
    
    assert df.F(1, 1) == df0.F(1, 1) *df.xi_DF(1, 1) # [N]