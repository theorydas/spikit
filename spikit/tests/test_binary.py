from spikit.binary import BlackHole, Binary
from spikit.units import pi

from spikit.tests.fixtures import default_binary
from pytest import approx
from numpy import array

# ============================
# ======== Mass Tests ========

def test_create_black_hole_mass():
    assert BlackHole(10).m == 10

def test_create_black_hole_associative():
    m_list = array([10, 20, 30]) # [Msun]
    
    assert sum(BlackHole(m_list).m) == BlackHole(sum(m_list)).m
    
def test_create_black_hole_radii():
    bh = BlackHole(10)
    
    assert bh.Rs == bh.Risco/3 # [Msun]
    
def test_calculate_total_mass():
    system = Binary(10, 5)
    
    assert system.m() == 15 # [Msun]

def test_calculate_mass_ratio():
    system = Binary(1e4, 2)
    
    assert system.q() == 0.0002

# =============================
# ======== Orbit Tests ========
    
def test_calculate_semiliatus_rectum(default_binary: Binary):
    system = default_binary
    
    assert system.p(a = 1e4, e = 0.5) == 7500

def test_calculate_separation(default_binary: Binary):
    system = default_binary
    
    assert system.r2(a = 100, e = 0.45, theta = 0) == 55
    assert system.r2(a = 100, e = 0.45, theta = pi) == 145

def test_cycle_sma(default_binary: Binary):
    system = default_binary
    
    a = 10
    
    Eorb = system.Eorb(a)
    
    a_ = system.a(Eorb)
    
    assert a == a_

def test_cycle_eccentricity(default_binary: Binary):
    system = default_binary
    
    a = 10
    e = 0.1
    
    Eorb = system.Eorb(a)
    Lorb = system.Lorb(a, e)
    
    e_ = system.e(Eorb, Lorb)
    
    assert e == approx(e_)

def test_vis_visa_velocity(default_binary: Binary):
    system = default_binary
    
    a = 10; r2 = 8
    
    u2 = system.u2(r2 = r2, a = a)
    u = system.u(r2 = r2, a = a)
    q = system.q()
    
    assert u/u2 == approx(1 +q)