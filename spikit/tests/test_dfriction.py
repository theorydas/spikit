from spikit.binary import black_hole, binary
from spikit.spike import static_powerlaw
from spikit.forces import dfriction

def test_iso_accretion_force():
    m1 = black_hole(1000); m2 = black_hole(5)
    system = binary(m1, m2, 100)
    spike = static_powerlaw(m1, 7/3, 1e16)
    
    df = dfriction(system, spike)
    assert df.F(1, 1) == df.F0(1, 1) *df.xi_DF(1, 1) # [N]