from spikit.binary import Binary
from spikit.forces import GravitationalWaves, DynamicalFrictionIso, AccretionIso
from spikit.units import yr, day, pc
from spikit.solvers import DynamicSolver
from spikit.spike import StaticPowerLaw
from spikit.blueprints import VacuumMerger, SpikeDFMerger


from spikit.tests.fixtures import default_binary, default_spike
from pytest import approx
from numpy import sum

def test_zero_evolution(default_binary: Binary):
    assert default_binary.da_dt(dE_dt = 0, dm2_dt = 0, r2 = 50, a = 100) == 0 # [pc/s]
    assert default_binary.de_dt(dE_dt = 0, dL_dt = 0, dm2_dt = 0, r2 = 50, a = 100, e = 0.45)  == 0 # [1/s]

def test_vacuum_merger_time_order_2(default_binary: Binary):
    # Setup binary and gravitational wave losses.
    binary = default_binary
    gw = GravitationalWaves(binary)
    
    t_expected_merger = 8 *yr # The time at which the merger is expected to occur.
    a0 = VacuumMerger(binary).r(t_expected_merger) # Starting distance [pc]

    result = DynamicSolver(binary, loss = gw).solve(a0, h = 1e-2)
    
    assert result["t"][-1] == approx(t_expected_merger, rel = 1e-2)

def test_vacuum_merger_time_order_1(default_binary: Binary):
    # Setup binary and gravitational wave loss.
    binary = default_binary
    gw = GravitationalWaves(binary)
    
    t_expected_merger = 8 *yr # The time at which the merger is expected to occur.
    a0 = VacuumMerger(binary).r(t_expected_merger) # Starting distance [pc]
    
    result = DynamicSolver(binary, loss = gw).solve(a0, h = 1e-2, order = 1)
    
    assert result["t"][-1] == approx(t_expected_merger, rel = 1e-1)

def test_zero_density_evolution_on_solvers(default_binary: Binary):
    binary = default_binary
    spike = StaticPowerLaw(binary, 7/3, 0)

    gw = GravitationalWaves(binary)
    df = DynamicalFrictionIso(spike)
    acc = AccretionIso(spike)
    
    t_expected_merger = 8 *yr # The time at which the merger is expected to occur.
    a0 = VacuumMerger(binary).r(t_expected_merger) # Starting distance [pc]
    
    result_vacuum = DynamicSolver(binary, loss = gw).solve(a0, h = 1e-1, order = 1)
    result_spike = DynamicSolver(binary, loss = [gw, df, acc]).solve(a0, h = 1e-1, order = 1)
    
    assert sum(result_vacuum["t"] -result_spike["t"]) == approx(0, rel = 1e-1)

def test_reconstruct_published_merger_with_solver():
    """ According to the paper arXiv:2002.12811v2, VB, page 14,
    this merger should be sped up by 48 days in 5 years. This used the old Coulomb logarithm.
    """
    binary = Binary(1.4e4, 1.4)
    spike = StaticPowerLaw(binary, 7/3, rhosp = 226)

    gw = GravitationalWaves(binary)
    df = DynamicalFrictionIso(spike)
    df.b_eff = lambda r2, u, q: r2 *pc *(q)**(0.5) # For old Coulomb logarithm.
    
    result = DynamicSolver(binary, loss = [gw, df]).solve(a0 = VacuumMerger(binary).r(5 *yr), h = 1e-2)
    
    assert (5 *yr -result["t"][-1])/day == approx(48, rel = 1e-1)

def test_df_merger_time_vs_solver(default_spike: StaticPowerLaw):    
    spike = default_spike
    
    gw = GravitationalWaves(spike.binary)
    df = DynamicalFrictionIso(spike)
    
    t_vacuum_merger = 5 *yr
    a0 = VacuumMerger(spike.binary).r(t_vacuum_merger)
    t_df_merger = SpikeDFMerger(spike).t_to_c(a0)
    
    result = DynamicSolver(spike.binary, loss = [gw, df]).solve(a0, h = 1e-2, order = 1)
    
    assert result["t"][-1] == approx(t_df_merger, rel = 1e-1)

def test_zero_density_merger_time(default_binary: Binary):    
    spike = StaticPowerLaw(default_binary, 7/3, rhosp = 0)
    
    gw = GravitationalWaves(spike.binary)
    df = DynamicalFrictionIso(spike)
    
    a0 = 100 *spike.binary.Risco() # [pc]
    
    t_df_merger = SpikeDFMerger(spike).t_to_c(a0)
    t_vacuum_merger = VacuumMerger(spike.binary).t_to_c(a0)
    
    assert t_vacuum_merger == approx(t_df_merger, 1e-3)