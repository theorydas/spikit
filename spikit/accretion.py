from spikit.spike import spike
from spikit.binary import binary
from spikit.units import c, pi, pc, Mo

class accretion:
    """ Isotropic accretion. """
    
    def __init__(self, spike: spike, binary: binary, k: float = None):
        self._spike = spike
        self._binary = binary
        
        self._k = k # The ratio of the body's Swartzchild radius to its physical size.
    
    def csection(self, u: float) -> float:
        """ Returns the cross-section [m^2] of the spike. """
        
        rs = self._binary._M2.Rs # [pc]
        beta = u/c
        
        if self._k == None: # The proper case of a black hole.
            return 4 *pi *rs**2 *(1 +1/beta**2) *pc**2 # [m2] x4 and k = 1.
        else:
            return pi *rs**2 /self._k**2 *(1 +self._k/beta**2) *pc**2 # [m2]
    
    def bacc(self, u: float) -> float:
        """ Returns the impact parameter b [m] of the spike. """
        
        return (self.csection(u)/pi)**(0.5)
    
    # ====================
    def dm2_dt0(self, r2: float, u: float):
        """ Returns the mass accretion rate [Msun/yr] due to the spike. """
        rho_DM = self._spike.rho(r2)/pc**3 # [Msun/m3]
        
        return rho_DM *self.csection(u) *u # [Msun/s]
    
    def F0(self, r2: float, u: float):
        """ Returns the accretion darg-force [N] due to the spike. """
        
        return self.dm2_dt0(r2, u)/Mo *u # [N]
        
    def dm2_dt(self, r2: float, u: float):
        """ Returns the mass accretion rate [Msun/yr] due to the spike. """
        
        return self.dm2_dt0(r2, u) *self.xi_m(r2, u) # [Msun/s]
    
    def F(self, r2: float, u: float):
        """ Returns the accretion darg-force [N] due to the spike. """
        return self.F0(r2, u) *self.xi_acc(r2, u) # [N]
    
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