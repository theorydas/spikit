from spikit.binary import *

def test_total_mass():
    m1 = black_hole(10); m2 = black_hole(5)
    b = binary(m1, m2, 100)
    
    assert b.m == 15 # [Msun]

def test_mass_ratio():
    m1 = black_hole(1e4); m2 = black_hole(2)
    b = binary(m1, m2, 100)
    
    assert b.mu == 5000