from spikit.forces import Accretion, AccretionIso

from spikit.binary import Binary
from spikit.spike import Spike

from spikit.tests.fixtures import default_binary, default_spike

def test_csection(default_binary: Binary, default_spike: Spike):
    acc = AccretionIso(default_binary, default_spike)
    accN = AccretionIso(default_binary, default_spike, k = 1)
    
    assert acc.csection(1) == 4 *accN.csection(1) # [m2]

def test_iso_accretion_force(default_binary: Binary, default_spike: Spike):
    acc = AccretionIso(default_binary, default_spike)
    acc0 = Accretion(default_binary, default_spike)
    
    assert acc.F(1, 1) == acc0.F(1, 1) *acc.xi_acc(1, 1) # [N]
    
def test_iso_accretion_mass(default_binary: Binary, default_spike: Spike):
    acc = AccretionIso(default_binary, default_spike)
    acc0 = Accretion(default_binary, default_spike)
    
    assert acc.dm2_dt(1, 1) == acc0.dm2_dt(1, 1) *acc.xi_m(1, 1) # [N]