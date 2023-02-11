from spikit.spike import StaticPowerLaw, PowerLaw
from spikit.binary import Binary
from pytest import approx

from spikit.tests.fixtures import default_binary, default_spike, default_dynaspike

def test_rho_conversion(default_spike: StaticPowerLaw):
    rho6 = 1
    
    rhosp = default_spike.rho_other(rho6 = rho6)
    rho6_ = default_spike.rho_other(rhosp = rhosp)
    
    assert rho6 == approx(rho6_, 1e-5)

def test_static_rho_initial(default_spike: StaticPowerLaw):    
    assert default_spike.rho(1) == default_spike.rho_init(1)

def test_static_feps_initial(default_spike: StaticPowerLaw):    
    assert default_spike.feps[0] == default_spike.feps_init(default_spike.eps)[0]

def test_static_fv_initial(default_spike: StaticPowerLaw):    
    assert default_spike.fv(1, 100) == default_spike.fv_init(1, 100)

def test_xi_no_particles(default_spike: StaticPowerLaw):
    assert default_spike.xi_Nl(0, chi = 0) == 0

def test_xi_any_k_order(default_spike: StaticPowerLaw):
    chi = 0.25
    assert default_spike.xi_Nu(0, chi = chi) == default_spike.xi_Nl(0, chi = 1) -default_spike.xi_Nl(0, chi = chi)
    assert default_spike.xi_Nu(1, chi = chi) == default_spike.xi_Nl(1, chi = 1) -default_spike.xi_Nl(1, chi = chi)
    assert default_spike.xi_Nu(2, chi = chi) == default_spike.xi_Nl(2, chi = 1) -default_spike.xi_Nl(2, chi = chi)

def test_density_reconstruction_from_distribution(default_binary: Binary, default_dynaspike: PowerLaw):
    r = 100 *default_binary.Risco()
    rho_analytical = default_dynaspike.rho_init(r)
    rho_numerical = default_dynaspike.rho(r)
    
    assert rho_analytical == approx(rho_numerical, 1e-4)

def test_density_reconstruction_with_velocity_range(default_binary: Binary, default_dynaspike: PowerLaw, default_spike: StaticPowerLaw):
    r = 100 *default_binary.Risco()
    chi_lower = 0.3
    chi_upper = 0.8
    
    rho_static = default_spike.rho(r, chi_lower = chi_lower, chi_upper = chi_upper)
    rho_dynamic = default_dynaspike.rho(r, chi_lower = chi_lower, chi_upper = chi_upper)
    
    assert rho_static == approx(rho_dynamic, 1e-3)

def test_velocity_moment_reconstruction(default_binary: Binary, default_dynaspike: PowerLaw, default_spike: StaticPowerLaw):
    r = 100 *default_binary.Risco()
    chi_max = 1
    k = 2 # Order of the moment
    Vmax = default_binary.Vmax(r)
    
    vk_static = default_spike.xi_Nl(N = k, chi = chi_max) *(chi_max *Vmax)**k
    vK_dynamic = default_dynaspike.v_moment(r, k = k, chi_lower = 0, chi_upper = chi_max)
    
    assert vk_static == approx(vK_dynamic, 1e-3)
    
def test_normalized_velocity_zero_moment(default_binary: Binary, default_dynaspike: PowerLaw):
    r = 100 *default_binary.Risco()
    
    assert default_dynaspike.v_moment(r, k = 0) == approx(1, 1e-3)

def test_velocity_distribution(default_binary: Binary, default_dynaspike: PowerLaw):
    r = 100 *default_binary.Risco()
    Vmax = default_binary.Vmax(r)
    v = 0.4 *Vmax
        
    assert default_dynaspike.fv(v, r) == approx(default_dynaspike.fv_init(v, r), 1e-3)