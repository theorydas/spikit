from spikit.units import G, c, Mo, pi, pc
from spikit.binary import Binary
from spikit.spike import Spike
from spikit.forces import DynamicalFrictionIso

from abc import ABC
from scipy.special import hyp2f1

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

class SpikeDFMerger(Blueprint):
    def __init__(self, spike: Spike, isotropic: bool = True):
        self._spike = spike
        self._binary = spike.binary
        self.df = DynamicalFrictionIso(spike)
        self.isotropic = isotropic # Whether to use the xi factor in the DF calculation.
    
    def t_to_c(self, r: float, m1: float = None, m2: float = None) -> float:
        if m1 is None: m1 = self._binary.m1
        if m2 is None: m2 = self._binary.m2
        
        m = m1 +m2 # [Msun]
        gammasp = self._spike.gammasp
        
        # This equation is only defined when lnL and xi are constant in space.
        # Pick a random point, i.e. 100Risco to calculate this.
        r2 = 100 *self._binary.Risco() # [pc]
        u = self._binary.u(r2) # [m/s]
        
        lnL = self.df.lnL(r2, u, m1, m2)
        xi = 1 if not self.isotropic else self.df.xi_DF(r2, u, m1)
        
        cGW = 64 *G**3 *m *m1 *m2 *Mo**3/(5 *c**5)
        cDF = 8 *pi *m2/m1 *(G/m/Mo)**(0.5) *lnL *self._spike.rho6 *Mo/pc**3 *(1e-6 *pc)**gammasp *xi
        
        ksi = cDF/cGW
        
        y = lambda r: 1/4 *(r *pc)**4 *hyp2f1(1, 8/(11 -2 *gammasp), 1 +8/(11 -2 *gammasp), -ksi *(r *pc)**(11/2 -gammasp))
        t = -(y(self._binary.Risco()) -y(r))/cGW
        
        return t # [s]