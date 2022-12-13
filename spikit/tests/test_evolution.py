from spikit.binary import black_hole, binary

def test_zero_evolution():
    m1 = black_hole(10); m2 = black_hole(5)
    system = binary(m1, m2, 100)

    assert system.da_dt(0, 0, 50, 100) == 0 # [pc/s]
    assert system.de_dt(0, 0, 0, 50, 100, 0.45)  == 0 # [1/s]