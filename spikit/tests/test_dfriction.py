from spikit.binary import BlackHole, Binary
from spikit.spike import StaticPowerLaw
from spikit.forces import DynamicalFriction, DynamicalFrictionIso

def test_iso_accretion_force():
    m1 = BlackHole(1000); m2 = BlackHole(5)
    system = Binary(m1, m2, 100)
    spike = StaticPowerLaw(m1, 7/3, 1e16)
    
    df = DynamicalFrictionIso(system, spike)
    df0 = DynamicalFriction(system, spike)
    
    assert df.F(1, 1) == df0.F(1, 1) *df.xi_DF(1, 1) # [N]