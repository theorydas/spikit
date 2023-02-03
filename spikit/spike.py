from spikit.binary import Binary
from spikit.units import Mo, pc, c, G

from scipy.special import gamma, hyp2f1
from warnings import warn
from abc import ABC
import numpy as np

class Spike(ABC):
    """ The default spike class. """
    def __init__(self, binary: Binary = None) -> None:
        self.binary = binary

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
        
        #  The normalization of the power law distribution function.
        self.A_f = 4/np.sqrt(np.pi) *gamma(self.gammasp +1)/gamma(self.gammasp -1/2)
    
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
    
    def xi_Nl(self, N: float, chi: float) -> float:
        """ Returns the lower, normalized velocity moment (v/u)^N for particles with v < u.
        
            chi is the ratio of the orbital velocity to the escape velocity.
            N is the order of the velocity-moment.
        """
        
        return self.A_f *chi**3 / (N +3) *hyp2f1(3/2 -self.gammasp, (N +3)/2, (N +5)/2, chi**2)

    def xi_Nu(self, N: float, chi: float) -> float:
        """ Returns the upper, normalized velocity moment (v/u)^N for particles with v > u.
        
            chi is the ratio of the orbital velocity to the escape velocity.
            N is the order of the velocity-moment.
        """
        
        return self.xi_Nl(N, 1) - self.xi_Nl(N, chi)