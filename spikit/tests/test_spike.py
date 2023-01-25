from spikit.spike import StaticPowerLaw
from spikit.binary import Binary
from pytest import approx

from spikit.tests.fixtures import default_binary, default_spike

def test_rho_conversion(default_spike: StaticPowerLaw):
    rho6 = 1
    
    rhosp = default_spike.rho_other(rho6 = rho6)
    rho6_ = default_spike.rho_other(rhosp = rhosp)
    
    assert rho6 == approx(rho6_, 1e-5)

def test_static_rho_initial(default_spike: StaticPowerLaw):
    rho = default_spike.rho(1)
    rho0 = default_spike.rho_init(1)
    
    assert rho == rho0

def test_no_particles(default_spike: StaticPowerLaw):
    assert default_spike.xi_Nl(0, chi = 0) == 0

def test_any_particles(default_spike: StaticPowerLaw):
    assert default_spike.xi_Nu(0, chi = 0.5) == 1 -default_spike.xi_Nl(0, chi = 0.5)