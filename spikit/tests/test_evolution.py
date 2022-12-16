from spikit.binary import BlackHole, Binary

def test_zero_evolution():
    m1 = BlackHole(10); m2 = BlackHole(5)
    system = Binary(m1, m2, 100)

    assert system.da_dt(0, 0, 50, 100) == 0 # [pc/s]
    assert system.de_dt(0, 0, 0, 50, 100, 0.45)  == 0 # [1/s]