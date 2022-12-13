from spikit.binary import black_hole, binary
from spikit.spike import static_powerlaw
from spikit.accretion import accretion

def test_csection():
    m1 = black_hole(1000); m2 = black_hole(5)
    system = binary(m1, m2, 100)
    spike = static_powerlaw(m1, 7/3, 1e16)
    
    acc = accretion(spike, system)
    accN = accretion(spike, system, 1)

    assert acc.csection(1) == 4 *accN.csection(1) # [m2]