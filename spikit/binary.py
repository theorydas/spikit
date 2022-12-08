from spikekit.units import *
import numpy as np

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
    
class binary:
    """ A binary of two black holes at a separation r2 at the largests ISCO radius. """
    
    def __init__(self, m1: black_hole, m2: black_hole, r2: float):
        
        if m1.m < m2.m: raise ValueError("m1 must be greater than m2.")
        if r2 < 1: raise ValueError("r2 must be greater than 1.")
        
        self._M1 = m1 # Largest black hole object.
        self._M2 = m2 # Smallest black hole object.
        
        self.r2 = r2 *self._M1.Risco # [pc]
    
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
    
    @property
    def rhill(self, r2: float = None) -> float:
        """" The Hill radius [pc] of the binary at its current separation, or at a given separation r2 [pc]. """
        if r2 is None: r2 = self.r2
        
        return self.r2 *(self.q/3)**(1/3) # [pc]
    
    # ======== Orbital Properties ========
    @property
    def Vmax(self, r2: float = None) -> float:
        """ The maximum orbital velocity [m/s] of the binary at its current separation, or at a given separation r2 [pc]. """
        if r2 is None: r2 = self.r2
        
        return np.sqrt(2 *G *self.m *Mo /(self.r2 *pc)) # [m/s]
    
    @property
    def uorb(self, r2: float = None) -> float:
        """ The orbital velocity [m/s] of the binary at its current position, or at a given separation r2 [pc]. """
        if r2 is None: r2 = self.r2
        
        return np.sqrt(G *self.m *Mo /(self.r2 *pc)) # [m/s]