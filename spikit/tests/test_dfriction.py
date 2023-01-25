from spikit.forces import DynamicalFriction, DynamicalFrictionIso
from spikit.spike import Spike

from spikit.tests.fixtures import default_spike

def test_iso_df_force(default_spike: Spike):
    df = DynamicalFrictionIso(default_spike)
    df0 = DynamicalFriction(default_spike)
    
    assert df.F(1, 1) == df0.F(1, 1) *df.xi_DF(1, 1) # [N]