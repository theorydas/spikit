from spikit.units import G, c, Mo, pi, pc
from spikit.binary import Binary
from spikit.spike import Spike, PowerLaw
from spikit.forces import DynamicalFrictionIso, Accretion
from spikit.feedback import AccretionDepletion

from abc import ABC
from scipy.special import hyp2f1, gamma, factorial, beta
import numpy as np
from copy import deepcopy

# A collection of analytical results for various specific (Blueprints) cases to serve as a reference.

class Merger(ABC):
    def __init__(self):
        pass
    
    def r(self, t: float, m1: float = None, m2: float = None) -> float:
        pass
    
    def t_to_c(self, r: float, m1: float = None, m2: float = None) -> float:
        pass
    
    def Phi(self, fGW: float, m1: float = None, m2: float = None) -> float:
        pass

class VacuumMerger(Merger):
    def __init__(self, binary: Binary):
        self._binary = binary
    
    def r(self, t: float, m1: float = None, m2: float = None) -> float:
        """ Returns the distance between the two objects [m] at time t. """
        
        if m1 is None: m1 = self._binary.m1
        if m2 is None: m2 = self._binary.m2
        
        m = m1 +m2 # [Msun]
        
        cGW = 64 *G**3 *m *m1 *m2 *Mo**3/(5 *c**5)
        risco = self._binary.Risco() *pc # [m]
        
        r = ( 4 *(t *cGW +risco**4/4) )**(1/4)
        
        return r/pc # [pc]
    
    
    def t_to_c(self, r: float, m1: float = None, m2: float = None) -> float:
        """ Returns the time it takes for the two objects to reach the ISCO separation [pc]. """
        
        if m1 is None: m1 = self._binary.m1
        if m2 is None: m2 = self._binary.m2
        
        m = m1 +m2 # [Msun]
        
        cGW = 64 *G**3 *m *m1 *m2 *Mo**3/(5 *c**5)
        
        y = lambda r: 1/4 *(r *pc)**4
        t = -(y(self._binary.Risco()) -y(r))/cGW
        
        return t # [s]
    
    def Phi(self, fGW: float, m1: float = None, m2: float = None) -> float:
        """ Calculates the phase related to a binary or gravitational wave signal."""
        
        if m1 is None: m1 = self._binary.m1
        if m2 is None: m2 = self._binary.m2
        
        Mc = self._binary.Mchirp(m1, m2) # [Msun]
        
        return 1/16 *(c**3 / (pi *G *Mc *Mo *fGW))**(5/3)

class SpikeDFMerger(Merger):
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
        u = self._binary.u2(r2) # [m/s]
        
        lnL = self.df.lnL(r2, u, m1, m2)
        xi = 1 if not self.isotropic else self.df.xi_DF(r2, u, m1)
        
        cGW = 64 *G**3 *m *m1 *m2 *Mo**3/(5 *c**5)
        cDF = 8 *pi *m2/m1 *(G/m/Mo)**(0.5) *lnL *self._spike.rho6 *Mo/pc**3 *(1e-6 *pc)**gammasp *xi
        
        ksi = cDF/cGW
        
        y = lambda r: 1/4 *(r *pc)**4 *hyp2f1(1, 8/(11 -2 *gammasp), 1 +8/(11 -2 *gammasp), -ksi *(r *pc)**(11/2 -gammasp))
        t = -(y(self._binary.Risco()) -y(r))/cGW
        
        return t # [s]

class StuckAccretionDepletedPowerLaw(StaticPowerLaw):
    """ A blueprint that describes what happens to the spike when the binary is 'stuck' at a given separation.
    
    The results are derived for a quasi-circular orbit and in absence of gravitational waves, external forces,
    but the feedback of accretion.
    """
    
    def __init__(self, accretion_force: Accretion):
        self.force = accretion_force
        self._spike = deepcopy(accretion_force._spike)
        self._binary = accretion_force._binary
    
    def rho(self, r: float, r2: float, t: float, Naccuracy = 1e-3) -> float:
        """ Returns the density of the spike at a given radius and time. """
        
        if Naccuracy <= 0: raise ValueError("Naccuracy must be greater than 0.")
        if t < 0: raise ValueError("t must be greater than 0.")
        if t == 0: return self._spike.rho_init(r)
        
        # ======
        rho6 = self._spike.rho6 # [Mo/pc3]
        gammasp = self._spike.gammasp
        r6 = 1e-6 # [pc]
        
        m1 = self._binary.m1 # [Msun]
        bacc = self.force.b_acc(self._binary.u2(r2)) /pc # [pc]
        T = self._binary.T(r2) # [s]
        
        # A constant that is used in the integral.
        A = 2/pi**0.5 *rho6 *(r6)**gammasp *gamma(gammasp +1)/gamma(gammasp -1/2) *(G *m1)**(-gammasp)
        
        # The original integral can be expanded in a power series, and each term split into two parts.
        # As long as the final density is not almost depleted, the convergence is stable.
        summation = 0
        n = 1
        while True:
            factor_n = 1/factorial(n) *( -r2 *bacc**2 /(G *m1)**3 *8 *t/T )**n
            
            P = G *m1/r
            P2 = G *m1/r2
            
            if r < r2:
                term = P**0.5 * P2**(3 *n +gammasp -1/2)
                term *= beta(gammasp -1/2 +5 *n/2, 1 +n/2)
                term *= hyp2f1(-1/2, gammasp -1/2 +5 *n/2, 3 *n +gammasp +1/2, P2/P)
            else:
                term = P**(gammasp +5/2 *n) *P2**(n/2)
                term *= beta(gammasp -1/2 +5 *n/2, 3/2)
                term *= hyp2f1(-n/2, gammasp -1/2 +5 *n/2, 5/2 *n +gammasp +1, P/P2)
            
            summation += factor_n *term
            n += 1

            if Naccuracy >= 1 and n > Naccuracy:
                break
            elif Naccuracy < 1 and abs(factor_n *term/summation) < Naccuracy:
                break
        
        return self._spike.rho_init(r) +A *summation
    
    def feps(self, r2: float, t: float) -> float:
        """ Returns the distribution function of the spike at a given radius and time. """
        
        u = self._binary.u2(r2) # [m/s]
        T = self._binary.T(r2) # [s]
        
        Pacc = AccretionDepletion(self.force).Pacc(r2, u)
        
        return self._spike.feps * np.exp( -Pacc *t/T)