from abc import ABC, abstractmethod
from spikit.units import c, pi, pc, Mo, G
from spikit.binary import Binary
from spikit.spike import Spike
from numpy import log


class Force(ABC):
    """ The base class for all forces acting on the companion of a binary. """
    
    def __init__(self):
        pass
    
    @abstractmethod
    def F(self, r2: float, u: float):
         pass
    
    def dE_dt(self, r2: float, u: float):
        """ Returns the rate of change of the orbital energy [J/s] due to the force. """
        return self.F(r2, u) *u # [J/s]
    
    def dlnL_dt(self, r2: float, u: float):
        """ Returns the rate of change of the log of the orbital angular momentum [log(J m/s)] due to the force. """
        return self.F(r2, u) /(self._binary.mu *u) # [log(J m/s)]
    
    def dL_dt(self, r2: float, u: float, a: float, e: float):
        """ Returns the rate of change of the orbital angular momentum [J m/s] due to the force. """
        return self.dlnL_dt(r2, u) *self._binary.Lorb(a, e) # [J m/s]

class Accretion(Force):
    def __init__(self, binary: Binary, spike: Spike, k: float = None):
        self._binary = binary
        self._spike = spike
        
        self._k = k # The ratio of the body's Swartzchild radius to its physical size.
    
    def csection(self, u: float) -> float:
        """ Returns the cross-section [m^2] of the spike. """
        
        rs = self._binary._M2.Rs # [pc]
        beta = u/c
        
        if self._k == None: # The proper case of a black hole.
            return 4 *pi *rs**2 *(1 +1/beta**2) *pc**2 # [m2] x4 and k = 1.
        else:
            return pi *rs**2 /self._k**2 *(1 +self._k/beta**2) *pc**2 # [m2]
    
    def b_acc(self, u: float) -> float:
        """ Returns the impact parameter b [m] of the spike. """
        
        return (self.csection(u)/pi)**(0.5)
    
    # ====================
    def dm2_dt(self, r2: float, u: float):
        """ Returns the mass accretion rate [Msun/yr] due to a 'frozen' spike. """
        rho_DM = self._spike.rho(r2)/pc**3 # [Msun/m3]

        return rho_DM *Accretion.csection(self, u) *u # [Msun/s]
    
    def F(self, r2: float, u: float):
        """ Returns the accretion darg-force [N] due to a 'frozen' spike. """
        
        return Accretion.dm2_dt(self, r2, u) *Mo *u # [N]

class DynamicalFriction(Force):
    """ The isotropic dynamical friction force. """
    
    def __init__(self, binary: Binary, spike: Spike):
        self._binary = binary
        self._spike = spike
    
    def b_90(self, r2: float, u: float):
        """ Returns the impact parameter [m] for which a particle is deflected by 90 degrees. """
        
        return G *self._binary.m2 *Mo/u**2 # [m]
    
    def b_eff(self, r2: float, u: float):
        """ Returns the maximum impact parameter [m] for which dynamical friction is effective. """
        
        return self._binary.rhill(r2) *pc # [m]
    
    def lnL(self, r2: float, u: float):
        """ Returns the standard Coulomb logarithm for the dynamical friction force. """
        b_eff = self.b_eff(r2, u) # [m] The effective impact parameter of the force.
        b_90 = self.b_90(r2, u) # [m]
        
        return log(1 +b_eff**2/b_90**2)
    
    def F(self, r2: float, u: float):
        """ Returns the dynamical friction force [N] due to a 'frozen' spike. """
        
        rho_DM = self._spike.rho(r2)/pc**3 # [Msun/m3]
        lnL = self.lnL(r2, u)
        
        return 4 *pi *G**2 *(self._binary.m2 *Mo)**2 *lnL *(rho_DM *Mo/pc**3)/u**2 # [N]

class AccretionIso(Accretion):
    """ The isotropic accretion force. """
        
    def dm2_dt(self, r2: float, u: float):
        """ Returns the mass accretion rate [Msun/yr] due to the spike. """
        
        return super().dm2_dt(r2, u) *self.xi_m(r2, u) # [Msun/s]
    
    def F(self, r2: float, u: float):
        """ Returns the accretion darg-force [N] due to the spike. """
        
        return super().F(r2, u) *self.xi_acc(r2, u) # [N]
    
    # ====================
    def xi_m(self, r2: float, u: float) -> float:
        """ Returns the boost to the mass accretion rate due to the spike. """
        
        k = self._k or 1
        chi = u/self._binary.Vmax(r2)
        beta = u/c
        
        xi_n1u = self._spike.xi_Nu(-1, chi)
        xi_0l = self._spike.xi_Nl(0, chi)
        xi_1u = self._spike.xi_Nu(1, chi)
        xi_2l = self._spike.xi_Nl(2, chi)
        
        return xi_0l +( xi_n1u *(k +beta**2 /3) +beta**2 *xi_1u + beta**2 /3 *xi_2l )/(beta**2 +k)

    def xi_acc(self, r2: float, u: float) -> float:
        """ Returns the aboost to the accretion rate due to the spike. """
        
        k = self._k or 1
        chi = u/self._binary.Vmax(r2)
        beta = u/c
        
        xi_n1u = self._spike.xi_Nu(-1, chi)
        xi_0l = self._spike.xi_Nl(0, chi)
        xi_1u = self._spike.xi_Nu(1, chi)
        xi_2l = self._spike.xi_Nl(2, chi)
        xi_4l = self._spike.xi_Nl(4, chi)
        
        return xi_0l +( xi_n1u *(10 *k + 4 *beta**2) +20 *beta**2 *xi_1u + xi_2l *(10 *beta**2 -5 *k) - beta**2 *xi_4l)/(beta**2 +k)/15

class DynamicalFrictionIso(DynamicalFriction): 
    def F(self, r2: float, u: float):
        """ Returns the dynamical friction force [N] due to a spike. """
        
        return super().F(r2, u) *self.xi_DF(r2, u) # [N]
    
    def xi_DF(self, r2: float, u: float) -> float:
        """ Returns the boost to the dynamical friction force due to the spike. """
        
        chi = u/self._binary.Vmax(r2)
        xi_0l = self._spike.xi_Nl(0, chi)
        
        return xi_0l