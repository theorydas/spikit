from spikit.binary import Binary
from spikit.forces import GravitationalWaves, DynamicalFrictionIso, AccretionIso
from spikit.units import yr
from spikit.solvers import StaticSolver
from spikit.spike import StaticPowerLaw


from spikit.tests.fixtures import default_binary
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