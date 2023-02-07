from spikit.forces import Accretion, AccretionIso
from spikit.spike import Spike, StaticPowerLaw
from spikit.binary import Binary
from spikit.feedback import AccretionDepletion
from spikit.blueprints import StuckAccretionDepletedPowerLaw

from spikit.tests.fixtures import default_binary, default_spike
from numpy import max, abs

def test_csection(default_spike: Spike):
    acc = AccretionIso(default_spike)
    accN = AccretionIso(default_spike, k = 1)
    
    assert acc.csection(1) == 4 *accN.csection(1) # [m2]

def test_iso_accretion_force(default_spike: Spike):
    acc = AccretionIso(default_spike)
    acc0 = Accretion(default_spike)
    
    assert acc.F(1, 1) == acc0.F(1, 1) *acc.xi_acc(1, 1) # [N]
    
def test_iso_accretion_mass(default_spike: Spike):
    acc = AccretionIso(default_spike)
    acc0 = Accretion(default_spike)
    
    assert acc.dm2_dt(1, 1) == acc0.dm2_dt(1, 1) *acc.xi_m(1, 1) # [N]

def test_zero_desnity_accretion_force(default_binary: Binary):
    spike = StaticPowerLaw(default_binary, 7/3, rho6 = 0)
    
    acc = Accretion(spike)
    
    assert acc.F(1, 1) == 0 # [N]

def test_zero_desnity_accretion_rate(default_binary: Binary):
    spike = StaticPowerLaw(default_binary, 7/3, rho6 = 0)
    
    acc = Accretion(spike)
    
    assert acc.dm2_dt(1, 1) == 0 # [N]

def test_stuck_acrretion_depletion_blueprint(default_binary: Binary, default_spike: Spike):
    r2 = 100 *default_binary.Risco() # [pc]
    T = default_binary.T(r2) # [s]
    u = default_binary.u2(r2) # [m/s]
    
    acc = Accretion(default_spike)
    acc.csection = lambda u: 1e16 # [m2]
    dfacc = AccretionDepletion(acc)
    blueprint = StuckAccretionDepletedPowerLaw(acc)
    
    t = 0
    for i in range(100):
        dt = 5 *T/100
        dfacc.dfeps_dt(r2, u)
        default_spike.f_eps += dfacc.dfeps_dt(r2, u) *dt
        t += dt
    
    assert max(abs(default_spike.f_eps/blueprint.feps(r2, t) -1)) <= 1e-2