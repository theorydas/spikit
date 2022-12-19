from spikit.binary import Binary
from spikit.forces import GravitationalWaves
from spikit.units import yr


from spikit.tests.fixtures import default_binary
from pytest import approx

def test_zero_evolution():
    system = Binary(10, 5)

    assert system.da_dt(dE_dt = 0, dm2_dt = 0, r = 50, a = 100) == 0 # [pc/s]
    assert system.de_dt(dE_dt = 0, dL_dt = 0, dm2_dt = 0, r = 50, a = 100, e = 0.45)  == 0 # [1/s]

def test_vacuum_merger_time(default_binary: Binary):
    def update(a: float, t: float, h: float = 1e-2) -> tuple:
        # First rugne-kutta step
        r2 = a; u = system.u(r2)
        
        dEdt_1 = -gw.dE_dt(r2, 0)
        dadt_1 = system.da_dt(dEdt_1, 0, r2, a) # [pc/s]

        dt = abs(a/dadt_1) *h

        a += 2/3 *dadt_1 *dt # [pc]
        
        # Second rugne-kutta step
        r2 = a; u = system.u(r2)
        
        dEdt_2 = -gw.dE_dt(r2, 0)
        dadt_2 = system.da_dt(dEdt_2, 0, r2, a) # [pc/s]
        
        t += dt
        a += dt/12 *(9 *dadt_2 -5 *dadt_1)
        
        return a, t
    
    # Setup binary and gravitational wave losses.
    system = default_binary
    gw = GravitationalWaves(system)
    
    t_expected_merger = 8 *yr # The time at which the merger is expected to occur.
    a0 = gw.vacuum_merger_distance(t_expected_merger) # Starting distance [pc]

    # Evolve the system.
    a_list = [a0]
    t_list = [0]

    while a_list[-1] > system.Risco():
        a = a_list[-1]; t = t_list[-1]

        a_, t_ = update(a, t_list[-1], h = 1e-2)
        a_list.append(a_); t_list.append(t_)

    assert t_ == approx(t_expected_merger, rel = 1e-2)