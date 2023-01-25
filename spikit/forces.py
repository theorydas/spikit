from abc import ABC
from spikit.units import c, pi, pc, Mo, G
from spikit.binary import Binary, BlackHole
from spikit.spike import Spike
from numpy import log

class Force(ABC):
    """ The base class for all forces acting on the companion of a binary. """
    
    def __init__(self, spike: Spike):
        self._binary = spike.binary
        self._spike = spike
    
    def dE_dt(self, r2: float, u: float, m1: float = None, m2: float = None) -> float:
        """ Returns the rate of change of the orbital energy [J/s] due to the force. """
        
        if m1 is None: m1 = self._binary.m1 # [Msun]
        if m2 is None: m2 = self._binary.m2 # [Msun]
        
        return self.F(r2, u, m1, m2) *u # [J/s]
    
    def dlnL_dt(self, r2: float, u: float, m1: float = None, m2: float = None) -> float:
        """ Returns the rate of change of the log of the orbital angular momentum [log(J m/s)] due to the force. """
        if m1 is None: m1 = self._binary.m1 # [Msun]
        if m2 is None: m2 = self._binary.m2 # [Msun]
        
        mu = Binary(m1, m2).mu() # [Msun]
        
        return self.F(r2, u, m1, m2) /(mu *Mo *u) # [log(J m/s)]
    
    def dL_dt(self, r2: float, u: float, a: float, e: float, m1: float = None, m2: float = None) -> float:
        """ Returns the rate of change of the orbital angular momentum [J m/s] due to the force. """
        
        if m1 is None: m1 = self._binary.m1 # [Msun]
        if m2 is None: m2 = self._binary.m2 # [Msun]
        
        return self.dlnL_dt(r2, u, m1, m2) *self._binary.Lorb(a, e, m1, m2) # [J m/s]

class Accretion(Force):
    """ The base accretion force. """
    
    def __init__(self, spike: Spike, k: float = None):
        if k == 0: raise ValueError("The parameter 'k' cannot be zero i.e. size of the object cannot be infinite.")
        
        super().__init__(spike)
        
        self._k = k # The ratio of the body's Swartzchild radius to its physical size.
    
    def csection(self, u: float, m2: float = None) -> float:
        """ Returns the cross-section [m^2] of the spike. """
        
        if m2 is None: m2 = self._binary.m2 # [Msun]
        
        rs = BlackHole(m2).Rs # [pc]
        beta = u/c
        
        if self._k == None: # The proper case of a black hole.
            return 4 *pi *rs**2 *(1 +1/beta**2) *pc**2 # [m2] x4 and k = 1.
        else:
            return pi *rs**2 /self._k**2 *(1 +self._k/beta**2) *pc**2 # [m2]
    
    def b_acc(self, u: float, m2: float = None) -> float:
        """ Returns the impact parameter b [m] of the spike. """
        
        if m2 is None: m2 = self._binary.m2 # [Msun]
        
        return (self.csection(u, m2)/pi)**(0.5)
    
    # ====================
    def dm2_dt(self, r2: float, u: float, m2: float = None) -> float:
        """ Returns the mass accretion rate [Msun/yr] due to a 'frozen' spike. """
        
        if m2 is None: m2 = self._binary.m2 # [Msun]
        
        rho_DM = self._spike.rho(r2)/pc**3 # [Msun/m3]

        return rho_DM *Accretion.csection(self, u, m2) *u # [Msun/s]
    
    def F(self, r2: float, u: float, m1: float = None, m2: float = None) -> float:
        """ Returns the accretion darg-force [N] due to a 'frozen' spike. """
        
        if m1 is None: m1 = self._binary.m1 # [Msun]
        if m2 is None: m2 = self._binary.m2 # [Msun]
        
        return Accretion.dm2_dt(self, r2, u, m2) *Mo *u # [N]

class DynamicalFriction(Force):
    """ The base dynamical friction force. """
    
    def b_90(self, r2: float, u: float, m2: float = None) -> float:
        """ Returns the impact parameter [m] for which a particle is deflected by 90 degrees. """
        
        if m2 == None: m2 = self._binary.m2 # [Msun]
        
        return G *m2 *Mo/u**2 # [m]
    
    def b_eff(self, r2: float, u: float, q: float = None) -> float:
        """ Returns the maximum impact parameter [m] for which dynamical friction is effective. """
        
        if q == None: q = self._binary.q
        
        return self._binary.rhill(r2, q) *pc # [m]
    
    def lnL(self, r2: float, u: float, m1: float = None, m2: float = None) -> float:
        """ Returns the standard Coulomb logarithm for the dynamical friction force. """
        
        if m1 is None: m1 = self._binary.m1 # [Msun]
        if m2 is None: m2 = self._binary.m2 # [Msun]
        
        b_eff = self.b_eff(r2, u, m2/m1) # [m] The effective impact parameter of the force.
        b_90 = self.b_90(r2, u, m2) # [m]
        
        return log(1 +b_eff**2/b_90**2)/2
    
    def F(self, r2: float, u: float, m1: float = None, m2: float = None) -> float:
        """ Returns the dynamical friction force [N] due to a 'frozen' spike. """
        
        if m1 is None: m1 = self._binary.m1 # [Msun]
        if m2 is None: m2 = self._binary.m2 # [Msun]
        
        rho_DM = self._spike.rho(r2) # [Msun/pc3]
        lnL = self.lnL(r2, u, m1, m2)
        
        return 4 *pi *G**2 *(m2 *Mo)**2 *lnL *(rho_DM *Mo/pc**3)/u**2 # [N]

# ====================

class AccretionIso(Accretion):
    """ The isotropic accretion force. """
        
    def dm2_dt(self, r2: float, u: float, m1: float = None, m2: float = None) -> float:
        """ Returns the mass accretion rate [Msun/yr] due to the spike. """
        
        if m1 is None: m1 = self._binary.m1 # [Msun]
        if m2 is None: m2 = self._binary.m2 # [Msun]
        
        return super().dm2_dt(r2, u) *self.xi_m(r2, u, m1) # [Msun/s]
    
    def F(self, r2: float, u: float, m1: float = None, m2: float = None) -> float:
        """ Returns the accretion darg-force [N] due to the spike. """
        
        if m1 is None: m1 = self._binary.m1 # [Msun]
        if m2 is None: m2 = self._binary.m2 # [Msun]
        
        return super().F(r2, u, m1, m2) *self.xi_acc(r2, u, m1) # [N]
    
    def xi_m(self, r2: float, u: float, m1: float = None) -> float:
        """ Returns the boost to the mass accretion rate due to the spike. """
        
        if m1 is None: m1 = self._binary.m1 # [Msun]
        
        k = self._k or 1
        chi = u/self._binary.Vmax(r2, m1)
        beta = u/c
        
        xi_n1u = self._spike.xi_Nu(-1, chi)
        xi_0l = self._spike.xi_Nl(0, chi)
        xi_1u = self._spike.xi_Nu(1, chi)
        xi_2l = self._spike.xi_Nl(2, chi)
        
        return xi_0l +( xi_n1u *(k +beta**2 /3) +beta**2 *xi_1u + beta**2 /3 *xi_2l )/(beta**2 +k)

    def xi_acc(self, r2: float, u: float, m1: float = None) -> float:
        """ Returns the aboost to the accretion rate due to the spike. """
        
        if m1 is None: m1 = self._binary.m1 # [Msun]
        
        k = self._k or 1
        chi = u/self._binary.Vmax(r2, m1)
        beta = u/c
        
        xi_n1u = self._spike.xi_Nu(-1, chi)
        xi_0l = self._spike.xi_Nl(0, chi)
        xi_1u = self._spike.xi_Nu(1, chi)
        xi_2l = self._spike.xi_Nl(2, chi)
        xi_4l = self._spike.xi_Nl(4, chi)
        
        return xi_0l +( xi_n1u *(10 *k + 4 *beta**2) +20 *beta**2 *xi_1u + xi_2l *(10 *beta**2 -5 *k) - beta**2 *xi_4l)/(beta**2 +k)/15

class DynamicalFrictionIso(DynamicalFriction): 
    """ The isotropic dynamical friction force. """
    
    def F(self, r2: float, u: float, m1: float = None, m2: float = None) -> float:
        """ Returns the dynamical friction force [N] due to a spike. """
        
        if m1 is None: m1 = self._binary.m1 # [Msun]
        if m2 is None: m2 = self._binary.m2 # [Msun]
        
        return super().F(r2, u, m1, m2) *self.xi_DF(r2, u, m1) # [N]
    
    def xi_DF(self, r2: float, u: float, m1: float = None) -> float:
        """ Returns the boost to the dynamical friction force due to the spike. """
        
        if m1 is None: m1 = self._binary.m1 # [Msun]
        
        chi = u/self._binary.Vmax(r2, m1)
        xi_0l = self._spike.xi_Nl(0, chi)
        
        return xi_0l

# ====================

class GravitationalWaves:
    def __init__(self, binary: Binary):
        self._binary = binary
    
    def dE_dt(self, a: float, u: float = 0, e: float = 0, m1: float = None, m2: float = None) -> float:
        """ Calculate the orbit averaged energy loss due to gravitational wave emission of an
        orbit with eccentricity e and semi major axis a per Eq. 15 of 2112.09586v1.
        
        * a the semi major axis in pc.
        """
        if m1 is None: m1 = self._binary.m1
        if m2 is None: m2 = self._binary.m2
        
        m = self._binary.m(m1, m2) # [Msun]
        mu = self._binary.mu(m1, m2) # [Msun]
        
        dEdt = 32/5 *mu**2 *m**3/(a *pc)**5 *G**4 / c**5 *Mo**5
        dEdt *= (1 +73/24 *e**2 +37/96 *e**4) *(1 -e**2)**(-7/2) # Weight because of non-zero eccentricity.
        
        return dEdt

    def dL_dt(self, a: float, e: float, m1: float = None, m2: float = None) -> float:
        """ Calculate the orbit averaged angular momentum loss due to gravitational wave emission of an
        orbit with eccentricity e and semi major axis a per Eq. 16 of 2112.09586v1.
        
        * a the semi major axis in pc.
        """
        if m1 is None: m1 = self._binary.m1
        if m2 is None: m2 = self._binary.m2
        
        m = self._binary.m(m1, m2) # [Msun]
        mu = self._binary.mu(m1, m2) # [Msun]
        
        dLdt = 32/5 *(mu *Mo)**2 *(m *Mo)**(5/2)/(a *pc)**(7/2) *G**(7/2) / c**5
        dLdt *= (1 +7/8*e**2) /(1 -e**2)**2 # Weight because of non-zero eccentricity.
        
        return dLdt
    
    # ======== Vacuum Systems ========
    #TODO: Must include the effect of the binary's ISCO into the time and ditstance calculations.
    def vacuum_merger_distance(self, t: float, m1: float = None, m2: float = None) -> float:
        if m1 is None: m1 = self._binary.m1
        if m2 is None: m2 = self._binary.m2
        
        return (256 * G**3 *(m1 +m2) *m1 *m2 *Mo**3 / (5 *c**5) *t)**(1/4) /pc # [pc]
    
    def vacuum_merger_time(self, r: float, m1: float = None, m2: float = None) -> float:
        if m1 is None: m1 = self._binary.m1
        if m2 is None: m2 = self._binary.m2
        
        return (5 *c**5 *(r *pc)**4) / (256 * G**3 *(m1 +m2) *m1 *m2 *Mo**3) # [s]
    
    def vacuum_phase(self, fGW: float, m1: float = None, m2: float = None) -> float:
        """ Calculates the phase related to a binary or gravitational wave signal."""
        
        if m1 is None: m1 = self._binary.m1
        if m2 is None: m2 = self._binary.m2
        
        Mc = self._binary.Mchirp(m1, m2) # [Msun]
        
        return 1/16 *(c**3 / (pi *G *Mc *Mo *fGW))**(5/3)