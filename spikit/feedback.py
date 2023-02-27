from abc import ABC
import numpy as np

from spikit.forces import Accretion
from spikit.units import pc, pi

class Feedback(ABC):
    def __init__(self, accretion_force: Accretion) -> None:
        self.force = accretion_force
        self.spike = accretion_force._spike
        self.binary = accretion_force._binary
    
    def dfeps_dt():
        pass

class AccretionDepletion(Feedback):
    def dfeps_dt(self, a: float, e: float = 0, N: int = 50, order: int = 1) -> float:
        
        if order not in [1, 2]: raise ValueError("Order must be 1 or 2")
        if N <= 0: raise ValueError("N must be positive")
        if e < 0 or e >= 1: raise ValueError("The eccentricity must be in the range [0, 1)")
        
        return -self.spike.f_eps *self.Pacc(a, e, N, order) /self.binary.T(a)
    
    def dgacc_dtheta(self, a: float, e: float = 0, theta: float = 0, order: int = 1) -> float:
        r2 = self.binary.r2(a, e, theta) # [pc]
        u = self.binary.u2(r2, a) # [m/s]
        
        if order == 1:
            csection = self.force.csection(u, r2) # [m2]
            vE = np.sqrt(2 *(self.binary.M1.Psi(r2) -self.spike.eps) ) # [m/s]
            
            gacc = 4 *pi *vE *(r2 *pc) *csection
        elif order == 2: # A more accurate approximation
            gacc = 1
        
        return np.nan_to_num(gacc)
        
    def Pacc(self, a: float, e: float = 0, N: int = 50, order: int = 1) -> float:
        """ The accretion probability """
        
        if order not in [1, 2]: raise ValueError("Order must be 1 or 2")
        if N <= 0: raise ValueError("N must be positive")
        if e < 0 or e >= 1: raise ValueError("The eccentricity must be in the range [0, 1)")
        
        theta_grid = np.linspace(0, pi, N).reshape(-1, 1) if e > 0 else 0 # This is half a circle due to symmetry, we need to multiply by 2 later.
        dgacc_dtheta = self.dgacc_dtheta(a, e, theta_grid, order) # A 2D array (theta, eps), 1D if e == 0
        
        g = self.spike.DoS(self.spike.eps)
        
        # If the orbit is circular, the integration is trivial, i.e. 2pi.
        if e == 0: return 2 *np.pi *np.nan_to_num(dgacc_dtheta/g)
        
        # Otherwise, we need to perform the integration around the orbit.
        gacc = 2 *np.trapz(dgacc_dtheta, theta_grid, axis = 0) # A 1D array (eps)
        
        return np.nan_to_num(gacc/g)