from spikit.binary import BlackHole, Binary
from spikit.units import pi

# ============================
# ======== Mass Tests ========

def test_calculate_total_mass():
    m1 = BlackHole(10); m2 = BlackHole(5)
    system = Binary(m1, m2)
    
    assert system.m == 15 # [Msun]

def test_calculate_mass_ratio():
    m1 = BlackHole(1e4); m2 = BlackHole(2)
    system = Binary(m1, m2)
    
    assert system.q == 0.0002

# =============================
# ======== Orbit Tests ========
    
def test_calculate_semiliatus_rectum():
    m1 = BlackHole(1e4); m2 = BlackHole(2)
    system = Binary(m1, m2)
    
    assert system.p(a = 1e4, e = 0.5) == 7500

def test_calculate_separation():
    m1 = BlackHole(1e4); m2 = BlackHole(1)
    system = Binary(m1, m2)
    
    
    assert system.r2(a = 100, e = 0.45, theta = 0) == 55
    assert system.r2(a = 100, e = 0.45, theta = pi) == 145