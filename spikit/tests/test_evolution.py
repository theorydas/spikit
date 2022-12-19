from spikit.binary import BlackHole, Binary

def test_zero_evolution():
    system = Binary(10, 5)

    assert system.da_dt(dE_dt = 0, dm2_dt = 0, r = 50, a = 100) == 0 # [pc/s]
    assert system.de_dt(dE_dt = 0, dL_dt = 0, dm2_dt = 0, r = 50, a = 100, e = 0.45)  == 0 # [1/s]