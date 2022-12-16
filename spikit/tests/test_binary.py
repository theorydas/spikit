from spikit.binary import BlackHole, Binary
from spikit.units import pi

# ============================
# ======== Mass Tests ========

def test_calculate_total_mass():
    m1 = BlackHole(10); m2 = BlackHole(5)
    system = Binary(m1, m2, 100)
    
    assert system.m == 15 # [Msun]

def test_calculate_mass_ratio():
    m1 = BlackHole(1e4); m2 = BlackHole(2)
    system = Binary(m1, m2, 100)
    
    assert system.q == 0.0002

# =============================
# ======== Orbit Tests ========

def test_fetch_orbit_elements():
    a = 100; e = 0.45
    
    m1 = BlackHole(1e4); m2 = BlackHole(1)
    system = Binary(m1, m2, a, e)
    
    
    assert system.a == a
    assert system.e == e
    
def test_calculate_semiliatus_rectum():
    m1 = BlackHole(1e4); m2 = BlackHole(2)
    system = Binary(m1, m2, 1e4, 0.5)
    
    assert system.p(system.a, system.e) == 7500

def test_calculate_separation():
    m1 = BlackHole(1e4); m2 = BlackHole(1)
    system = Binary(m1, m2, 100, 0.45)
    
    
    assert system.r2(system.a, system.e, 0) == 55
    assert system.r2(system.a, system.e, pi) == 145