from spikit.binary import BlackHole, Binary
from spikit.spike import StaticPowerLaw
from spikit.forces import Accretion, AccretionIso

def test_csection():
    m1 = BlackHole(1000); m2 = BlackHole(5)
    system = Binary(m1, m2, 100)
    spike = StaticPowerLaw(m1, 7/3, 1e16)
    
    acc = AccretionIso(system, spike)
    accN = AccretionIso(system, spike, k = 1)

    assert acc.csection(1) == 4 *accN.csection(1) # [m2]

def test_iso_accretion_force():
    m1 = BlackHole(1000); m2 = BlackHole(5)
    system = Binary(m1, m2, 100)
    spike = StaticPowerLaw(m1, 7/3, 1e16)
    
    acc = AccretionIso(system, spike)
    acc0 = Accretion(system, spike)
    
    assert acc.F(1, 1) == acc0.F(1, 1) *acc.xi_acc(1, 1) # [N]
    
def test_iso_accretion_mass():
    m1 = BlackHole(1000); m2 = BlackHole(5)
    system = Binary(m1, m2, 100)
    spike = StaticPowerLaw(m1, 7/3, 1e16)
    
    acc = AccretionIso(system, spike)
    acc0 = Accretion(system, spike)
    
    assert acc.dm2_dt(1, 1) == acc0.dm2_dt(1, 1) *acc.xi_m(1, 1) # [N]