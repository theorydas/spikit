from spikit.binary import Binary
from spikit.forces import GravitationalWaves
from spikit.units import yr
from spikit.solvers import StaticSolver


from spikit.tests.fixtures import default_binary
from pytest import approx

def test_zero_evolution():
    system = Binary(10, 5)

    assert system.da_dt(dE_dt = 0, dm2_dt = 0, r = 50, a = 100) == 0 # [pc/s]
    assert system.de_dt(dE_dt = 0, dL_dt = 0, dm2_dt = 0, r = 50, a = 100, e = 0.45)  == 0 # [1/s]

def test_vacuum_merger_time(default_binary: Binary):
    # Setup binary and gravitational wave losses.
    binary = default_binary
    gw = GravitationalWaves(binary)
    
    t_expected_merger = 8 *yr # The time at which the merger is expected to occur.
    a0 = gw.vacuum_merger_distance(t_expected_merger) # Starting distance [pc]
    
    t, a = StaticSolver(binary, gw).solve(gw.vacuum_merger_distance(t_expected_merger), h = 1e-2)
    
    assert t[-1] == approx(t_expected_merger, rel = 1e-2)