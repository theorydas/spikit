from spikit.forces import Accretion, AccretionIso

from spikit.tests.fixtures import default_binary_spike_system

def test_csection(default_binary_spike_system):    
    acc = AccretionIso(*default_binary_spike_system)
    accN = AccretionIso(*default_binary_spike_system, k = 1)

    assert acc.csection(1) == 4 *accN.csection(1) # [m2]

def test_iso_accretion_force(default_binary_spike_system):  
    acc = AccretionIso(*default_binary_spike_system)
    acc0 = Accretion(*default_binary_spike_system)
    
    assert acc.F(1, 1) == acc0.F(1, 1) *acc.xi_acc(1, 1) # [N]
    
def test_iso_accretion_mass(default_binary_spike_system):    
    acc = AccretionIso(*default_binary_spike_system)
    acc0 = Accretion(*default_binary_spike_system)
    
    assert acc.dm2_dt(1, 1) == acc0.dm2_dt(1, 1) *acc.xi_m(1, 1) # [N]