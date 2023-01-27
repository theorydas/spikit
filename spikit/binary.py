from spikit.units import *
from numpy import sqrt, cos

# ============================
# ======== Black Hole ========
class BlackHole:
    """ A black hole with mass m [Msun]. """
    
    def __init__(self, m: float):
        self.m = m # [Msun]
    
    @property
    def Rm(self) -> float:
        """ The mass length-scale [pc] of the black hole. """
        
        return G *(self.m *Mo) /c**2 /pc # [pc]
    
    @property
    def Rs(self) -> float:
        """ The Swartzchild radius [pc] of the black hole. """
        
        return 2 *self.Rm # [pc]
    
    @property
    def Risco(self) -> float:
        """ The innermost stable circular orbit (ISCO) radius [pc] of the black hole. """
        
        return 6 *self.Rm # [pc]

# ========================
# ======== Binary ========
class Binary:
    """ A binary of two black holes at a separation r2 [pc] """
    
    def __init__(self, m1: float, m2: float):
        
        if m1 < m2: raise ValueError("m1 must be greater than m2.")
        
        self.m1 = m1 # [Msun] Central black hole.
        self.m2 = m2 # [Msun] Companion.
    
    def m(self, m1: float = None, m2: float = None) -> float:
        """ The total mass [Msun] of the binary. """
        if m1 is None: m1 = self.m1
        if m2 is None: m2 = self.m2
        
        return self.m1 +self.m2 # [Msun]
    
    def mu(self, m1: float = None, m2: float = None) -> float:
        """ The reduced mass [Msun] of the binary. """
        if m1 is None: m1 = self.m1
        if m2 is None: m2 = self.m2
        
        return self.m1 *self.m2/(self.m1 +self.m2) # [Msun]
    
    def Mchirp(self, m1: float = None, m2: float = None) -> float:
        """ The chirp mass [Msun] of the binary. """
        if m1 is None: m1 = self.m1
        if m2 is None: m2 = self.m2
        
        return (m1 *m2)**(3/5) / (m1 +m2)**(1/5) # [Msun]
    
    def q(self, m1: float = None, m2: float = None) -> float:
        """" The mass ratio of the binary. """
        if m1 is None: m1 = self.m1
        if m2 is None: m2 = self.m2
        
        return self.m2/self.m1
    
    def Risco(self, m: float = None) -> float:
        """ The innermost stable circular orbit (ISCO) radius [pc] of the binary. """
        if m is None: m = self.m1 # [Msun]
        
        return BlackHole(m).Risco # [pc]
    
    def rhill(self, r2: float, q: float = None) -> float:
        """" The Hill radius [pc] of the binary at a given separation r2 [pc]. """
        if q is None: q = self.q()
        
        return r2 *(q/3)**(1/3) # [pc]
    
    # ======== Orbital Properties ========
    
    def p(self, a: float, e: float) -> float:
        """ The semi-latus rectum [pc] of the binary. """
        
        return a *(1 -e**2) # [pc]
    
    def r2(self, a: float = None, e = 0, theta: float = 0) -> float:
        """ The separation [pc] of the binary at a given semi-major axis a [pc]. """
        
        return self.p(a, e) /(1 +e *cos(theta)) # [pc]
    
    def Vmax(self, r: float, m1: float = None) -> float:
        """ The maximum orbital velocity [m/s] around the larger black hole at a given radius r [pc]. """
        if m1 is None: m1 = self.m1
        
        return sqrt(2 *G *m1 *Mo /(r *pc)) # [m/s]
    
    def u(self, r2: float, a: float = None, m: float = None) -> float:
        """ The orbital velocity [m/s] of the binary at a given separation r2 [pc]
        and semi-major axis a [pc]. """
        
        if a is None: a = r2 # [pc]
        if m is None: m = self.m()
        
        return sqrt(G *self.m() *Mo *(2/r2 -1/a)/pc) # [m/s]
    
    def T(self, a: float, m: float = None) -> float:
        """ The orbital period [s] of the binary at a given semi-major axis a [pc]. """
        if m is None: m = self.m()
        
        return 2 *pi *sqrt((a *pc)**3 /(G *m *Mo)) # [s]
    
    def f(self, a: float, m: float = None) -> float:
        """ The orbital frequency [Hz] of the binary at a given semi-major axis a [pc]. """
        
        if m is None: m = self.m()
        
        return 1/self.T(a, m)
    
    def u1(self, r2: float, a: float = None, m1: float = None, m2: float = None) -> float:
        """ The orbital velocity [m/s] of the larger black hole at a given separation r2 [pc]. """
        
        if m1 is None: m1 = self.m1
        if m2 is None: m2 = self.m2
        if a is None: a = r2 # [pc]
        
        return self.u(r2, a) *m2/(m1 +m2) # [m/s]
    
    def u2(self, r2: float, a: float = None, m1: float = None, m2: float = None) -> float:
        """ The orbital velocity [m/s] of the smaller black hole at a given separation r2 [pc]. """
        
        if m1 is None: m1 = self.m1
        if m2 is None: m2 = self.m2
        if a is None: a = r2 # [pc]
        
        return self.u(r2, a) *m1/(m1 +m2) # [m/s]
    
    # ======== Conservative Quantities ========
    
    def Eorb(self, a: float, m1: float = None, m2: float = None) -> float:
        """ The (negative) orbital energy [J] of the binary at a given semi-major axis a [pc]. """
        
        if m1 is None: m1 = self.m1
        if m2 is None: m2 = self.m2
        
        return -G *self.m1 *self.m2 *Mo**2/(2 *a *pc) # [J]
    
    def Lorb(self, a: float, e: float = 0, m1: float = None, m2: float = None) -> float:
        """ The orbital angular momentum [Js] of the binary at a given semi-major axis a [pc] and eccentricity e. """
        if m1 is None: m1 = self.m1
        if m2 is None: m2 = self.m2
        
        m = self.m(m1, m2) # [Msun]
        mu = self.mu(m1, m2) # [Msun]
        
        return mu *Mo *sqrt(G *m *Mo *self.p(a, e) *pc)
    
    def a(self, Eorb: float, m1: float = None, m2: float = None) -> float:
        """ The semi-major axis [pc] of the binary at a given orbital energy Eorb [J]. """
        
        if m1 is None: m1 = self.m1
        if m2 is None: m2 = self.m2
        
        return -G *m1 *m2 *Mo**2/(2 *Eorb) /pc # [pc]
    
    def e(self, Eorb: float, Lorb: float, m1: float = None, m2: float = None) -> float:
        if m1 is None: m1 = self.m1
        if m2 is None: m2 = self.m2
        
        mu = self.mu(m1, m2) # [Msun]
        m = self.m(m1, m2) # [Msun]
        
        a = self.a(Eorb) # [pc] 
        p = (Lorb/mu/Mo)**2 / (G *m *Mo) /pc # [pc]
        
        return sqrt(1 -p/a)
    
    # ======== Evolution ========
    
    def da_dt(self, dE_dt: float, dm2_dt: float, r2: float, a: float, m: float = None) -> float:
        """ The rate of change of the semi-major axis [pc/s] given a force F [N] acting
        in the direction of its motion and a mass rate for the companion [Msun/s]. """
        
        if m is None: m = self.m() # [Msun]
        
        return -a *( dE_dt/self.Eorb(a) + dm2_dt/m *(2 *a/r2 -1)) # [pc/s]
    
    def de_dt(self, dE_dt: float, dL_dt: float, dm2_dt: float, r: float, a: float, e: float, m1: float = None) -> float:
        """ The rate of change of the eccentricity [1/s] given a force F [N] acting
        in the direction of its motion and a mass rate for the companion [Msun/s]. """
        
        if e == 0: return 0
        if m1 is None: m1 = self.m1 # [Msun]
        
        return - (1 -e**2)/e *( dE_dt/2/self.Eorb(a) + dL_dt/self.Lorb(a, e) + dm2_dt/m1 *(a/r -1)) # [1/s]