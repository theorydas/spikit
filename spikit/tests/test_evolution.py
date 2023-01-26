from spikit.binary import Binary
from spikit.forces import GravitationalWaves, DynamicalFrictionIso, AccretionIso
from spikit.units import yr, day, pc
from spikit.solvers import StaticSolver
from spikit.spike import StaticPowerLaw


from spikit.tests.fixtures import default_binary, default_spike
from pytest import approx
from numpy import sum

def test_zero_evolution():
    system = Binary(10, 5)

def test_vacuum_merger_time_order_2(default_binary: Binary):
    # Setup binary and gravitational wave losses.
    binary = default_binary
    gw = GravitationalWaves(binary)
    
    t_expected_merger = 8 *yr # The time at which the merger is expected to occur.
    a0 = gw.vacuum_merger_distance(t_expected_merger) # Starting distance [pc]
    
    t, a = StaticSolver(binary, gw).solve(gw.vacuum_merger_distance(t_expected_merger), h = 1e-2)
    
    assert t[-1] == approx(t_expected_merger, rel = 1e-2)

def test_vacuum_merger_time_order_1(default_binary: Binary):
    # Setup binary and gravitational wave losses.
    binary = default_binary
    gw = GravitationalWaves(binary)
    
    t_expected_merger = 8 *yr # The time at which the merger is expected to occur.
    a0 = gw.vacuum_merger_distance(t_expected_merger) # Starting distance [pc]
    
    t, a = StaticSolver(binary, gw).solve(gw.vacuum_merger_distance(t_expected_merger), h = 1e-2, order = 1)
    
    assert t[-1] == approx(t_expected_merger, rel = 1e-1)

def test_zero_density_evolution(default_binary: Binary):
    binary = default_binary
    spike = StaticPowerLaw(binary, 7/3, 0)

    gw = GravitationalWaves(binary)
    df = DynamicalFrictionIso(spike)
    acc = AccretionIso(spike)
    
    t_expected_merger = 8 *yr # The time at which the merger is expected to occur.
    a0 = gw.vacuum_merger_distance(t_expected_merger) # Starting distance [pc]
    
    t_vacuum, a_vacuum = StaticSolver(binary, gw).solve(gw.vacuum_merger_distance(t_expected_merger), h = 1e-1, order = 1)
    t_spike, a_spike = StaticSolver(binary, [gw, df, acc]).solve(a0, h = 1e-1, order = 1)
    
    assert sum(t_vacuum -t_spike) == approx(0, rel = 1e-1)

def test_reconstruct_published_merger():
    """ According to the paper arXiv:2002.12811v2, VB, page 14,
    this merger should be sped up by 48 days in 5 years. This used the old Coulomb logarithm.
    """
    binary = Binary(1.4e4, 1.4)
    spike = StaticPowerLaw(binary, 7/3, rhosp = 226)

    gw = GravitationalWaves(binary)
    df = DynamicalFrictionIso(spike)
    df.b_eff = lambda r2, u, q: r2 *pc *(q)**(0.5) # For old Coulomb logarithm.
    
    t, _ = StaticSolver(binary, [gw, df]).solve(gw.vacuum_merger_distance(5 *yr), h = 1e-2)
    
    assert (5 *yr -t[-1])/day == approx(48, rel = 1e-1)

def test_df_merger_time(default_spike: StaticPowerLaw):    
    spike = default_spike
    
    gw = GravitationalWaves(spike.binary)
    df = DynamicalFrictionIso(spike)
    
    t_vacuum_merger = 5 *yr
    a0 = gw.vacuum_merger_distance(t_vacuum_merger)
    t_df_merger = df.df_merger_time(a0)
    
    t, a = StaticSolver(spike.binary, [gw, df]).solve(a0, h = 1e-2, order = 1)
    
    assert t[-1] == approx(t_df_merger, rel = 1e-1)

def test_zero_density_merger_time(default_binary: Binary):    
    spike = StaticPowerLaw(default_binary, 7/3, rhosp = 0)
    
    gw = GravitationalWaves(spike.binary)
    df = DynamicalFrictionIso(spike)
    
    a0 = 100 *spike.binary.Risco() # [pc]
    
    t_df_merger = df.df_merger_time(a0)
    t_df_merger = gw.vacuum_merger_time(a0)
    
    assert t_df_merger == t_df_merger