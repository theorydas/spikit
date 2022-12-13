from spikit.units import *
import numpy as np

# ============================
# ======== Black Hole ========
class black_hole:
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
class binary:
    """ A binary of two black holes at a separation r2 at the largests ISCO radius. """
    
    def __init__(self, m1: black_hole, m2: black_hole, a: float, e: float = 0):
        
        if m1.m < m2.m: raise ValueError("m1 must be greater than m2.")
        if a < m1.Risco : raise ValueError("a must be greater than the ISCO radius.")
        
        self._M1 = m1 # Largest black hole object.
        self._M2 = m2 # Smallest black hole object.
        
        self.a = a # Semi-major axis [pc]
        self.e = e # Eccentricity.
    
    @property
    def m1(self) -> float:
        """ The mass [Msun] of the larger black hole. """
        return self._M1.m # [Msun]
    
    @property
    def m2(self) -> float:
        """ The mass [Msun] of the smaller black hole. """
        return self._M2.m # [Msun]
    
    @property
    def m(self) -> float:
        """ The total mass [Msun] of the binary. """
        return self.m1 +self.m2 # [Msun]
    
    @property
    def mu(self) -> float:
        """ The reduced mass [Msun] of the binary. """
        return self.m1 *self.m2/(self.m1 +self.m2) # [Msun]
    
    @property
    def q(self) -> float:
        """" The mass ratio of the binary. """
        return self.m2/self.m1
    
    @property
    def Risco(self) -> float:
        """ The innermost stable circular orbit (ISCO) radius [pc] of the binary. """
        return self._M1.Risco # [pc]
    
    def rhill(self, r2: float) -> float:
        """" The Hill radius [pc] of the binary at a given separation r2 [pc]. """
        
        return r2 *(self.q/3)**(1/3) # [pc]
    
    # ======== Orbital Properties ========
    
    def p(self, a: float, e: float):
        return a *(1 -e**2) # [pc]
    
    def r2(self, a: float = None, e = 0, theta: float = 0) -> float:
        """ The separation [pc] of the binary at a given semi-major axis a [pc]. """
        
        if a is None: a = self.a
        
        return self.p(a, e) /(1 +e *np.cos(theta)) # [pc]
    
    def Vmax(self, r: float) -> float:
        """ The maximum orbital velocity [m/s] around the larger black hole at a given radius r [pc]. """
        
        return np.sqrt(2 *G *self.m1 *Mo /(r *pc)) # [m/s]
    
    def u(self, r2: float, a: float) -> float:
        """ The orbital velocity [m/s] of the binary at a given separation r2 [pc]
        and semi-major axis a [pc]."""
        
        return np.sqrt(G *self.m *Mo *(2/r2 -1/a)/pc) # [m/s]