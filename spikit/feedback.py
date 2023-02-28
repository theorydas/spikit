from abc import ABC
import numpy as np

from spikit.forces import Accretion, Force
from spikit.units import pc, pi

from scipy.special import ellipeinc, ellipkinc

class Feedback(ABC):
    """ A class that describes feedback from the companion onto the spike. """
    
    def __init__(self, force: Force) -> None:
        self.force = force
        self.spike = force._spike
        self.binary = force._binary

class AccretionDepletion(Feedback):
    """ The depletion of the spike due to particle accretion on the companion."""
    def __init__(self, accretion_force: Accretion) -> None:
        super().__init__(accretion_force)
        
        self.ellipeinc = np.vectorize(self.ellipeinc)
        self.ellipkinc = np.vectorize(self.ellipkinc)
        
        self.I1 = lambda u, alpha: 2 *np.sqrt(1 +u) *self.ellipeinc(alpha/2, 2 *u/(1+u))
        self.I2 = lambda u, alpha: 2/3/u*( (u-1)*np.sqrt(1+u) *self.ellipkinc(alpha/2, 2*u/(1+u)) +u*np.sin(alpha) *np.sqrt(1+u*np.cos(alpha)) +np.sqrt(1+u) *self.ellipeinc(alpha/2, 2*u/(1+u)) )
    
    def dfeps_dt(self, a: float, e: float = 0, N: int = 50, order: int = 1) -> float:
        """ The time derivative of the distribution function due to particle accretion on the companion.
        
        Parameters
        ----------
            a: float, The semi-major axis [pc] of the orbit.
            e: float = 0, The eccentricity of the elliptic orbit.
            N: float = 50, The number of points to use in the integration around the orbit.
            order: int, The order of the approximation. 1 is the fastest, while 2 and 3 are different implementations of a more accurate approximation.
        """
        if order not in [1, 2, 3]: raise ValueError("Order must be 1, 2 or 3")
        if N <= 0: raise ValueError("N must be positive")
        if e < 0 or e >= 1: raise ValueError("The eccentricity must be in the range [0, 1)")
        
        return -self.spike.f_eps *self.Pacc(a, e, N, order) /self.binary.T(a)
    
    def dgacc_dtheta(self, a: float, e: float = 0, theta: float = 0, order: int = 1) -> float:
        """ The density of states that describes particles in the spike which are accreted onto the companion.
        
        Parameters
        ----------
            a: float, The semi-major axis [pc] of the orbit.
            e: float = 0, The eccentricity of the elliptic orbit.
            N: float = 50, The number of points to use in the integration around the orbit.
            order: int, The order of the approximation. 1 is the fastest, while 2 and 3 are different implementations of a more accurate approximation.
        """
        r2 = self.binary.r2(a, e, theta) # [pc]
        u = self.binary.u2(r2, a) # [m/s]
        
        Ep = self.spike.eps/self.binary.M1.Psi(r2)
        X = self.force.b_acc(u, r2)/(r2 *pc)
        
        if order == 1:
            csection = self.force.csection(u, r2) # [m2]
            vE = np.sqrt(2 *(self.binary.M1.Psi(r2) -self.spike.eps) ) # [m/s]
            
            gacc = 4 *pi *vE *(r2 *pc) *csection
        elif order == 2: # A more accurate approximation
            vE = np.sqrt(2 *(self.binary.M1.Psi(r2) -self.spike.eps) ) # [m/s]
            
            xs = np.linspace(0, X, 50)
            integrand = []
            for x in xs:
                u = x/(1 +Ep)
                alpha_max = np.arccos(np.clip(-1/u, -1, 1))
                integrand.append(x *self.I1(u, alpha_max) +x**2 *np.nan_to_num(self.I2(u, alpha_max)))
            
            gacc = 2 *4 *np.pi *(r2 *pc)**3 *vE *np.trapz(integrand, xs, axis = 0)
        elif order == 3: # An alternative implementation of 2.
            getIntegrant_ = lambda alpha: 2*(35*Ep*(1 - Ep)**(5/2) + 35*Ep*(Ep - 1)*(-Ep + X*np.cos(alpha) + 1)**(3/2) + 21*(1 - 2*Ep)*(1 - Ep)**(5/2) - 15*(1 - Ep)**(7/2) + (42*Ep - 21)*(-Ep + X*np.cos(alpha) + 1)**(5/2) + 15*(-Ep + X*np.cos(alpha) + 1)**(7/2))/(105*np.cos(alpha)**2)
            
            alphas = np.linspace(0, np.pi, 50)
            
            integrand = []
            for alpha_ in alphas:
                integrand_ = getIntegrant_(alpha_)
                integrand_[1 +X *np.cos(alpha_) -Ep < 0] = 0 # We remove points outside the domain.
                integrand.append(integrand_)
            
            gacc = 2 *4 *np.pi *(r2 *pc)**3 *np.sqrt(2 *self.binary.M1.Psi(r2)) *np.trapz(integrand, alphas, axis = 0)
        
        return np.nan_to_num(gacc)
        
    def Pacc(self, a: float, e: float = 0, N: int = 50, order: int = 1) -> float:
        """ The probability that particles in the spike are accreted onto the companion.
        
        Parameters
        ----------
            a: float, The semi-major axis [pc] of the orbit.
            e: float = 0, The eccentricity of the elliptic orbit.
            N: float = 50, The number of points to use in the integration around the orbit.
            order: int, The order of the approximation. 1 is the fastest, while 2 and 3 are different implementations of a more accurate approximation.
        """
        
        if order not in [1, 2, 3]: raise ValueError("Order must be 1, 2 or 3")
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
    
    def ellipkinc(self, x: float, m: float) -> float:
        """ An extension for the incomplete elliptic integral of the first kind that is also valid for m > 1.
        
        Parameters
        ----------
            x: float, The amplitude of the elliptic integral.
            m: float, The parameter of the elliptic integral, m = k^2.
        """
        if m <= 1:
            return ellipkinc(x, m)
        else: # Extend domain with Reciprocal modulus transformations.
            sinb = np.clip(np.sqrt(m) *np.sin(x), -1, 1)
            b = np.arcsin(sinb)
            return 1/np.sqrt(m) *ellipkinc(b, 1/m)
    
    def ellipeinc(self, x: float, m: float) -> float:
        """ An extension for the incomplete elliptic integral of the second kind that is also valid for m > 1.
        
        Parameters
        ----------
            x: float, The amplitude of the elliptic integral.
            m: float, The parameter of the elliptic integral, m = k^2.
        """
        if m <= 1:
            return ellipeinc(x, m)
        else: # Extend domain with Reciprocal modulus transformations.
            sinb = np.clip(np.sqrt(m) *np.sin(x), -1, 1)
            b = np.arcsin(sinb)
            return np.sqrt(m) *ellipeinc(b, 1/m) + (1 -m)/np.sqrt(m) *ellipkinc(b, 1/m)