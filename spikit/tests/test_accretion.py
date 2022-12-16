from spikit.binary import black_hole, binary
from spikit.spike import static_powerlaw
from spikit.forces import accretion

def test_csection():
    m1 = black_hole(1000); m2 = black_hole(5)
    system = binary(m1, m2, 100)
    spike = static_powerlaw(m1, 7/3, 1e16)
    
    acc = accretion(system, spike)
    accN = accretion(system, spike, k = 1)

    assert acc.csection(1) == 4 *accN.csection(1) # [m2]

def test_iso_accretion_force():
    m1 = black_hole(1000); m2 = black_hole(5)
    system = binary(m1, m2, 100)
    spike = static_powerlaw(m1, 7/3, 1e16)
    
    acc = accretion(system, spike)
    assert acc.F(1, 1) == acc.F0(1, 1) *acc.xi_acc(1, 1) # [N]
    
def test_iso_accretion_mass():
    m1 = black_hole(1000); m2 = black_hole(5)
    system = binary(m1, m2, 100)
    spike = static_powerlaw(m1, 7/3, 1e16)
    
    acc = accretion(system, spike)
    assert acc.dm2_dt(1, 1) == acc.dm2_dt0(1, 1) *acc.xi_m(1, 1) # [N]