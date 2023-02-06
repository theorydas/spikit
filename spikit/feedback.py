from abc import ABC
import numpy as np

from spikit.binary import Binary
from spikit.spike import Spike
from spikit.forces import Accretion
from spikit.units import pc

class Feedback(ABC):
    def __init__(self, accretion_force: Accretion) -> None:
        self.force = accretion_force
        self.spike = accretion_force._spike
        self.binary = accretion_force._binary
    
    def dfeps_dt():
        pass

class AccretionDepletion(Feedback):
    def dfeps_dt(self, r: float, v: float) -> float:
        """ The accretion rate [Mo/yr] """
        
        return -self.spike.f_eps *self.Pacc(r, v) /self.binary.T(r)
    
    def Pacc(self, r: float, u: float) -> float:
        """ The accretion power [W] """
        
        csection = self.force.csection(u) 
        vE = np.sqrt(2 *(self.binary.M1.Psi(r) -self.spike.eps) ) # m/s
        
        gacc = 8 *np.pi**2 *vE *(r *pc) *csection
        g = self.spike.DoS(self.spike.eps)
         
        return np.nan_to_num(gacc/g)