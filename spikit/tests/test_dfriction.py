from spikit.forces import DynamicalFriction, DynamicalFrictionIso
from spikit.spike import Spike, StaticPowerLaw
from spikit.binary import Binary

from spikit.tests.fixtures import default_binary, default_spike

def test_iso_df_force(default_spike: Spike):
    df = DynamicalFrictionIso(default_spike)
    df0 = DynamicalFriction(default_spike)
    
    assert df.F(1, 1) == df0.F(1, 1) *df.xi_DF(1, 1) # [N]

def test_zero_desnity_df_force(default_binary: Binary):
    spike = StaticPowerLaw(default_binary, 7/3, rho6 = 0)
    
    acc = DynamicalFriction(spike)
    
    assert acc.F(1, 1) == 0 # [N]