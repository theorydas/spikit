from spikit.binary import Binary
from spikit.forces import GravitationalWaves
from spikit.units import yr
from spikit.solvers import StaticSolver


from spikit.tests.fixtures import default_binary
from pytest import approx

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