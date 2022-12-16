from spikit.units import *
import numpy as np

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
    
    def __init__(self, M1: BlackHole, M2: BlackHole):
        
        if M1.m < M2.m: raise ValueError("m1 must be greater than m2.")
        
        self._M1 = M1 # Largest black hole object.
        self._M2 = M2 # Smallest black hole object.
    
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
    
    def p(self, a: float, e: float) -> float:
        """ The semi-latus rectum [pc] of the binary. """
        return a *(1 -e**2) # [pc]
    
    def r2(self, a: float = None, e = 0, theta: float = 0) -> float:
        """ The separation [pc] of the binary at a given semi-major axis a [pc]. """
        
        return self.p(a, e) /(1 +e *np.cos(theta)) # [pc]
    
    def Vmax(self, r: float) -> float:
        """ The maximum orbital velocity [m/s] around the larger black hole at a given radius r [pc]. """
        
        return np.sqrt(2 *G *self.m1 *Mo /(r *pc)) # [m/s]
    
    def u(self, r2: float, a: float = None) -> float:
        """ The orbital velocity [m/s] of the binary at a given separation r2 [pc]
        and semi-major axis a [pc]."""
        
        if a is None: a = r2 # [pc]
        
        return np.sqrt(G *self.m *Mo *(2/r2 -1/a)/pc) # [m/s]
    
    def T(self, a: float) -> float:
        """ The orbital period [s] of the binary at a given semi-major axis a [pc]. """
        
        return 2 *np.pi *np.sqrt((a *pc)**3 /(G *self.m *Mo)) # [s]
    
    def f(self, a: float) -> float:
        """ The orbital frequency [Hz] of the binary at a given semi-major axis a [pc]. """
        
        return 1/self.T(a)
    
    # ======== Conservative Quantities ========
    
    def Eorb(self, a: float) -> float:
        """ The (negative) orbital energy [J] of the binary at a given semi-major axis a [pc]. """
        
        return -G *self.m1 *self.m2 *Mo**2/(2 *a *pc) # [J]
    
    def Lorb(self, a: float, e: float = 0) -> float:
        """ The orbital angular momentum [Js] of the binary at a given semi-major axis a [pc] and eccentricity e. """
        
        return self.mu *Mo *np.sqrt(G *self.m *Mo *self.p(a, e) *pc)
    
    # ======== Evolution ========
    
    def da_dt(self, dE_dt: float, dm2_dt: float, r: float, a: float) -> float:
        """ The rate of change of the semi-major axis [pc/s] given a force F [N] acting
        in the direction of its motion and a mass rate for the companion [Msun/s]. """
        
        return -a *( dE_dt/self.Eorb(a) + dm2_dt/self.m1 *(2 *a/r -1)) # [pc/s]
    
    def de_dt(self, dE_dt: float, dL_dt: float, dm2_dt: float, r: float, a: float, e: float) -> float:
        """ The rate of change of the eccentricity [1/s] given a force F [N] acting
        in the direction of its motion and a mass rate for the companion [Msun/s]. """
        
        if e == 0: return 0
        
        return - (1 -e**2)/e *( dE_dt/2/self.Eorb(a) + dL_dt/self.Lorb(a, e) + dm2_dt/self.m1 *(a/r -1)) # [1/s]