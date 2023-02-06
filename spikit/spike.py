from spikit.binary import Binary
from spikit.units import Mo, pc, c, G

from scipy.special import gamma, hyp2f1
from warnings import warn
from abc import ABC
import numpy as np

N_GRID = 100_000 # The number of grid points used in the distribution function.

class Spike(ABC):
    """ The default spike class. """
    def __init__(self, binary: Binary = None) -> None:
        self.binary = binary
    
    def Psi(self, r):
        pass
    
    def fv(self, v: float, r: float) -> float:
        rho = self.rho(r) * Mo/pc**3 # [Kg/m3]
        
        feps = np.interp(self.Psi(r) -v**2/2, self.eps, self.f_eps)
        fv = 4 *np.pi *v**2/rho *feps
        
        return fv
    
    def rho(self, r: float, chi_min: float = 0, chi_max: float = 1) -> float:
        """ The density of the spike [Mo/pc3]. """
        if r == 0:
            warn("The spike's density is undefined at r = 0.")
            return 0
        
        v = np.linspace(chi_min, chi_max, 500)[::-1] *self.binary.Vmax(r)
        eps = self.Psi(r) - v**2/2
        
        feps = np.interp(eps, self.eps, self.f_eps)
        
        return 4 *np.pi *np.trapz(feps *v, eps) /(Mo/pc**3) # [Mo/pc3]

    def v_moment(self, r: float, k: float = 0, chi_min: float = 0, chi_max: float = 1) -> float:
        """ The k-th velocity moment of the spike. """
        
        v = np.linspace(chi_min, chi_max, 500) *self.binary.Vmax(r)
        fv = self.fv_init(v, r)
        
        return np.trapz(fv *v**k, v)
    
    def DoS(self, eps: float) -> float:
        """ The density of states that describes particles in the spike. """
        
        m1 = self.binary.m1
        return np.sqrt(2) *(np.pi *G *m1 *Mo)**3 *self.eps**(-5/2)
        

class PowerLaw(Spike):
    """ The default power-law spike. """
    
    def __init__(self, binary: Binary = None, gammasp: float = 7/3, rho6: float = 0, rhosp: float = 0):
        super().__init__(binary)
        
        self.gammasp = gammasp # Power-law index of the spike.
        self.rho6 = rho6
        self.rhosp = rhosp
        
        if rho6 != 0 and rhosp == 0:
            self.rhosp = self.rho_other(rho6 = rho6, gammasp = gammasp, m1 = binary.m1)
        elif rhosp != 0 and rho6 == 0:
            self.rho6 = self.rho_other(rhosp = rhosp, gammasp = gammasp, m1 = binary.m1)
        elif rho6 == 0 and rhosp == 0:
            warn("No spike density given, results MAY converge to vacuum case.")
        elif rho6 != 0 and rhosp != 0:
            raise ValueError("rho6 and rhosp are BOTH non-zero, please only specify one.")
        
        self.Psi = self.binary.M1.Psi
        self.r_min = self.binary.M1.Rs
        self.r_max = self.rsp()
        
        self.eps = self.Psi(np.logspace(np.log10(self.r_max), np.log10(self.r_min), N_GRID)) # The energy grid.
        self.f_eps = self.feps_init(self.eps) # The distribution function.
    
    def rsp(self, gammasp: float = None, m1: float = None, rhosp: float = None, rho6: float = None) -> float:
        """ The spike's size [pc].
        
        * gammasp is the slope of the power law.
        * m1 is the mass [M_sun] of the larger component in the binary.
        * rhosp [Mo/pc3] is the normalisation density of the spike.

        The calculations are based on Equation 2.2 of arxiv.org/abs/2002.12811.
        """
        
        if gammasp is None: gammasp = self.gammasp
        if m1 is None: m1 = self.binary.m1
        if rhosp is None and rho6 is None:
            rhosp = self.rhosp
        elif rhosp is None:
            rhosp = self.rho_other(rho6 = rho6, gammasp = gammasp, m1 = m1)
        
        if rhosp == 0:
            warn("No spike density given, the spike's size is `infinite` and set to 1 pc.")
            return 1 # [pc]
        
        return ((3 -gammasp) *(m1 *Mo)/(2 *np.pi *(rhosp *Mo/pc**3) *5**(3 -gammasp)))**(1/3) /pc
    
    def rho_other(self, rho6: float = 0, rhosp: float = 0, gammasp: float = None, m1: float = None) -> float:
        """ A conversion between the spike density normalisation rhosp [M_sun/pc3] and rho6 [M_sun/pc3].

        * rhosp/(rho6) is the density normalisation of the spike/(at distance r6 = 1e-6 pc).
        * gammasp is the slope of the density distribution.
        * m1 is the mass [M_sun] of the central black hole in the binary system.
        """
        
        if rho6 != 0 and rhosp != 0:
            raise ValueError("rho6 and rhosp are BOTH non-zero, please only specify one.")
        
        if gammasp is None: gammasp = self.gammasp
        if m1 is None: m1 = self.binary.m1
        
        if rho6 == 0:
            r6 = 1e-6 # [pc]
            rsp = self.rsp(gammasp, m1, rhosp = rhosp) # [pc]
            rho6 = rhosp *(r6/rsp)**-gammasp
            
            return rho6
        elif rhosp == 0:
            # Use this definition of r6 to avoid recursion.
            r6 = 1e-6 *pc # [m]
            A = ( (3 - gammasp) *0.2**(3-gammasp) *(m1 *Mo)/(2 *np.pi) )**(gammasp/3)
            rhosp = ((rho6 *Mo/pc**3)/A *r6 **gammasp)**(1/(1 -gammasp/3)) /(Mo/pc**3)
            
            return rhosp
    
    def rho_init(self, r: float) -> float:
        return self.rho6 *(r/1e-6)**-self.gammasp # [Msun/pc3]
    
    def feps_init(self, eps: float) -> float:
        r6 = 1e-6 *pc # [m]
        rho6 = self.rho6 *Mo/pc**3 # [kg/m3]
        gammasp = self.gammasp
        m1 = self.binary.m1
        
        N = rho6 *r6**(gammasp) *gamma(gammasp +1)/gamma(gammasp -1/2) *(G *m1 *Mo)**(-gammasp) /np.sqrt(2 *np.pi)**3
        
        return N *eps**(gammasp -3/2)
    
    def fv_init(self, v: float, r: float) -> float:
        Vmax = self.binary.Vmax(r)
        
        Av = 4/np.sqrt(np.pi) *gamma(self.gammasp +1)/gamma(self.gammasp -1/2)
        
        return Av *(1 -v**2/Vmax**2)**(self.gammasp -3/2) *v**2/Vmax**3 # [s/m]
    
    def f_break_static(self, m1: float = None, m2: float = None, gammasp: float = None, rhosp: float = None, rho6: float = None) -> float:
        """ Returns the break frequency [Hz] as defined by the matching of the gravitational
        and dynamic friction energy losses of a static power law dark matter spike in Equation 15
        of arxiv.org/pdf/2108.04154.pdf.

        * m1, m2 are the masses [M_sun] of the two components.
        * gammasp is the slope of the dark matter distribution.
        * rhosp is the density [M_sun/pc3] normalisation of the spike.
        """
        if gammasp is None: gammasp = self.gammasp
        if m1 is None: m1 = self.binary.m1
        if m2 is None: m2 = self.binary.m2
        if rhosp is None and rho6 is not None:
            rhosp = self.rho_other(rho6 = rho6, gammasp = gammasp, m1 = m1)
        elif rhosp is None: rhosp = self.rhosp
        
        lnL = np.log(m1/m2)/2
        ksi = self.xi_Nl(0, np.sqrt(2)/2)

        # The size of the spike
        rsp = self.rsp(gammasp, m1, rhosp) *pc # [m]

        c_f = (5 *c**5)/(8 *(m1 *Mo)**2) *np.pi**(2/3 *(gammasp -4)) *G**(-2/3 -gammasp/3)\
        * (m1 *Mo +m2 *Mo)**(1/3 -gammasp/3) *rsp**gammasp *ksi *(rhosp *Mo/(pc)**3) *lnL
        
        fb = c_f **(3/(11 -2 *gammasp)) # [Hz]
        
        return fb # [Hz]

class StaticPowerLaw(PowerLaw):
    """ An isotropic, power-law spike with a power-law index gammasp. """
    
    def rho(self, r: float, chi_min: float = 0, chi_max: float = 1) -> float:
        xi_DF = self.xi_Nl(0, chi_max) -self.xi_Nl(0, chi_min)
        
        return self.rho_init(r) *xi_DF # [Msun/pc3]
    
    def feps(self, eps: float) -> float:
        return self.feps_init(eps)
    
    def fv(self, v: float, r: float) -> float:
        return self.fv_init(v, r)
    
    def xi_Nl(self, N: float, chi: float) -> float:
        """ Returns the lower, normalized velocity moment (v/u)^N for particles with v < u.
        
            chi is the ratio of the velocity u to the escape velocity.
            N is the order of the velocity-moment.
        """
        A_v = 4/np.sqrt(np.pi) *gamma(self.gammasp +1)/gamma(self.gammasp -1/2)
        
        return A_v *chi**3 / (N +3) *hyp2f1(3/2 -self.gammasp, (N +3)/2, (N +5)/2, chi**2)

    def xi_Nu(self, N: float, chi: float) -> float:
        """ Returns the upper, normalized velocity moment (v/u)^N for particles with v > u.
        
            chi is the ratio of the velocity u to the escape velocity.
            N is the order of the velocity-moment.
        """
        
        return self.xi_Nl(N, 1) - self.xi_Nl(N, chi)