from spikit.units import G, c, Mo, pi, pc
from spikit.binary import Binary

from abc import ABC

class Blueprint(ABC):
    def __init__(self):
        pass
    
    def r(self, t: float, m1: float = None, m2: float = None) -> float:
        """ Returns the distance between the two objects [m] at time t. """
        
        if m1 is None: m1 = self._binary.m1
        
        pass
    
    def t_to_c(self, r: float, m1: float = None, m2: float = None) -> float:
        """ Returns the time it takes for the two objects to reach the ISCO separation [m]. """
        
        if m1 is None: m1 = self._binary.m1
    
    def Phi(self, fGW: float, m1: float = None, m2: float = None) -> float:
        """ Calculates the phase related to a binary or gravitational wave signal."""

        pass

class VacuumMerger(Blueprint):
    def __init__(self, binary: Binary):
        self._binary = binary
    
    #TODO: Must include the effect of the binary's ISCO into the time and ditstance calculations.
    def r(self, t: float, m1: float = None, m2: float = None) -> float:
        """ Returns the distance between the two objects [m] at time t. """
        
        if m1 is None: m1 = self._binary.m1
        if m2 is None: m2 = self._binary.m2
        
        return (256 * G**3 *(m1 +m2) *m1 *m2 *Mo**3 / (5 *c**5) *t)**(1/4) /pc # [pc]
    
    def t_to_c(self, r: float, m1: float = None, m2: float = None) -> float:
        """ Returns the time it takes for the two objects to reach the ISCO separation [m]. """
        
        if m1 is None: m1 = self._binary.m1
        if m2 is None: m2 = self._binary.m2
        
        return (5 *c**5 *r**4) / (256 * G**3 *(m1 +m2) *m1 *m2 *Mo**3) # [s]
    
    def Phi(self, fGW: float, m1: float = None, m2: float = None) -> float:
        """ Calculates the phase related to a binary or gravitational wave signal."""
        
        if m1 is None: m1 = self._binary.m1
        if m2 is None: m2 = self._binary.m2
        
        Mc = self._binary.Mchirp(m1, m2) # [Msun]
        
        return 1/16 *(c**3 / (pi *G *Mc *Mo *fGW))**(5/3)