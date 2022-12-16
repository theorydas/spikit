from spikit.binary import BlackHole, Binary

def test_zero_evolution():
    m1 = BlackHole(10); m2 = BlackHole(5)
    system = Binary(m1, m2)

    assert system.da_dt(dE_dt = 0, dm2_dt = 0, r = 50, a = 100) == 0 # [pc/s]
    assert system.de_dt(dE_dt = 0, dL_dt = 0, dm2_dt = 0, r = 50, a = 100, e = 0.45)  == 0 # [1/s]