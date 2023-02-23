from spikit.binary import Binary
from spikit.units import Mo, pc, c, G

from scipy.special import gamma, hyp2f1
from warnings import warn
from abc import ABC
import numpy as np

N_GRID = 100_000 # The number of grid points used in the distribution function.

class Spike(ABC):
    """ The base class for all spikes. It is an abstract class that handles their distribution functions. """
  
    def __init__(self, m1: float, m2: float = 0) -> None:
        self.binary = Binary(m1, m2)
        self.eps = 0
        self.f_eps = 0
    
    def Psi(self, r: float) -> float:
        """ The gravitational potential [J] that is bounding the spike. """
        pass
    
    def fv(self, v: float, r: float) -> float:
        """ The 1D-velocity distribution [s/m] of the spike at a given distance [pc].
        This is obtained from the distribution function, and is used to calculate the velocity moments.
        
        Parameters
        ----------
            v: float, The velocity magnitude [m/s] of the particle.
            r: float, The distance [pc] from the centre of the spike where the distribution is evaluated.
        """
        rho = self.rho(r) * Mo/pc**3 # [Kg/m3]
        
        feps = np.interp(self.Psi(r) -v**2/2, self.eps, self.f_eps)
        fv = 4 *np.pi *v**2/rho *feps
        
        return fv
    
    def rho(self, r: float, v_lower: float = 0, v_upper: float = None, chi_lower: float = 0, chi_upper: float = 1) -> float:
        """ The density [Msun/pc3] of the spike at a given distance [pc] from the centre, and within a given velocity range.
        The range can be specified by the ratio of the velocity to the maximum velocity of the spike, chi.
        Alternatively, the range can be specified by the velocity itself.
        
        Parameters
        ----------
            r: float, The distance [pc] from the centre of the spike where the density is evaluated.
            v_lower: float = 0, The upper velocity [m/s] limit.
            v_upper: float = None, The lower velocity [m/s] limit.
            chi_lower: float = 0, The ratio of the upper velocity limit to the maximum velocity of the spike.
            chi_upper: float = 1, The ratio of the lower velocity limit to the maximum velocity of the spike.
        """
        if r <= 0: warn("Distance must be positive. Returning 0."); return 0
        
        Vmax = self.binary.Vmax(r) # [m/s]
        if chi_lower < 0 or v_lower < 0:
            chi_lower = 0; v_lower = 0
            warn("The lower velocity limit should not be negative. Proceed as if zero.")
        
        if chi_upper > 1 or v_lower > Vmax:
            chi_lower = 1; v_lower = Vmax
            warn("The upper velocity limit should not exceed escape velocity. Proceed as if at escape.")
        
        if chi_lower > 0:
            if v_lower > 0: raise ValueError("Only one of v_lower or chi_lower should be given.")
            v_lower = chi_lower *Vmax # [m/s]
        
        if chi_upper < 1:
            if v_upper is not None: raise ValueError("Only one of v_upper or chi_upper should be given.")
            v_upper = chi_upper *Vmax # [m/s]
        
        if v_upper is None: v_upper = chi_upper *Vmax # [m/s]
        if v_lower == v_upper: return 0 # If the range has no width, don't do the calculation.
        
        v = np.linspace(v_lower, v_upper, 500)[::-1] # [m/s]
        eps = self.Psi(r) - v**2/2 # [m2/s2]
        
        feps = np.interp(eps, self.eps, self.f_eps)
        
        return 4 *np.pi *np.trapz(feps *v, eps) /(Mo/pc**3) # [Msun/pc3]

    def v_moment(self, r: float, k: float = 0, v_lower: float = 0, v_upper: float = None, chi_lower: float = 0, chi_upper: float = 1) -> float:
        """ The k-th 1D-velocity moment of the particle distribution for particles within a velocity range.
        For example, for k = 0, the default calculation should give the distribution's normalization (about 1),
        while for k = 1, it is the average velocity of all particles.
        The range can be specified by the ratio of the velocity to the maximum velocity of the spike, chi.
        Alternatively, the range can be specified by the velocity itself.
        
        Parameters
        ----------
            r: float, The distance [pc] from the centre of the spike where the density is evaluated.
            k: float, The index k of the velocity moment.
            v_lower: float = 0, The upper velocity [m/s] limit.
            v_upper: float = None, The lower velocity [m/s] limit.
            chi_lower: float = 0, The ratio of the upper velocity limit to the maximum velocity of the spike.
            chi_upper: float = 1, The ratio of the lower velocity limit to the maximum velocity of the spike.
        """
        if r <= 0: warn("Distance must be positive. Returning 0."); return 0
        
        Vmax = self.binary.Vmax(r) # [m/s]
        if chi_lower < 0 or v_lower < 0:
            chi_lower = 0; v_lower = 0
            warn("The lower velocity limit should not be negative. Proceed as if zero.")
        
        if chi_upper > 1 or v_lower > Vmax:
            chi_lower = 1; v_lower = Vmax
            warn("The upper velocity limit should not exceed escape velocity. Proceed as if at escape.")
        
        if chi_lower > 0:
            if v_lower > 0: raise ValueError("Only one of v_lower or chi_lower should be given.")
            v_lower = chi_lower *Vmax # [m/s]
        
        if chi_upper < 1:
            if v_upper is not None: raise ValueError("Only one of v_upper or chi_upper should be given.")
            v_upper = chi_upper *Vmax # [m/s]
        
        if v_upper is None: v_upper = chi_upper *Vmax # [m/s]
        if v_lower == v_upper: return 0 # If the range has no width, don't do the calculation.
        
        v = np.linspace(v_lower, v_upper, 500) # [m/s]
        fv = self.fv_init(v, r)
        
        return np.trapz(fv *v**k, v)
    
    def DoS(self, eps: float = None) -> float:
        """ The density of states (DoS) [m4/s] describing a particle with specific energy eps [m2/s2] in the spike.
        When no eps is given, it uses the spike's eps grid instead.
        
        Parameters
        ----------
            eps: float = self.eps, The specific energy [m2/s2] of the particles in the dark matter spike.
        """
        if eps is None: eps = self.eps # [m2/s2]
        
        m1 = self.binary.m1
        return np.sqrt(2) *(np.pi *G *m1 *Mo)**3 *eps**(-5/2)

class PowerLaw(Spike):
    """ A spike with a PowerLaw density distribution around a binary system with index gammasp, and normalization rho6 [Msun/pc3] at 1e-6 [pc].
    Alternatively to rho6, it can be initialized using the the density rhosp [Msun/pc3] at its size.
    """
    
    def __init__(self, m1: float, m2: float = 0, gammasp: float = 7/3, rho6: float = 0, rhosp: float = 0):
        super().__init__(m1, m2)
        
        self.gammasp = gammasp # Power-law index of the spike.
        self.rho6 = rho6
        self.rhosp = rhosp
        self.binary = Binary(m1, m2)
        
        if rho6 != 0 and rhosp == 0:
            self.rhosp = self.rho_other(rho6 = rho6, gammasp = gammasp, m1 = self.binary.m1)
        elif rhosp != 0 and rho6 == 0:
            self.rho6 = self.rho_other(rhosp = rhosp, gammasp = gammasp, m1 = self.binary.m1)
        elif rho6 == 0 and rhosp == 0:
            warn("No spike density given, results MAY converge to vacuum case.")
        elif rho6 != 0 and rhosp != 0:
            raise ValueError("rho6 and rhosp are BOTH non-zero, please only specify one.")
        
        self.Psi = self.binary.M1.Psi
        self.r_min = self.binary.M1.Rs
        self.r_max = self.rsp()
        
        self.eps = self.Psi(np.logspace(np.log10(self.r_max), np.log10(self.r_min), N_GRID)) # The energy grid.
        self.f_eps = self.feps_init(self.eps) # The distribution function.
    
    @classmethod
    def from_binary(cls, binary: Binary, gammasp: float = 7/3, rho6: float = 0, rhosp: float = 0):
        return cls(binary.m1, binary.m2, gammasp, rho6, rhosp)
    
    def rsp(self, gammasp: float = None, m1: float = None, rhosp: float = None, rho6: float = None) -> float:
        """ The spike's size [pc]. This calculation is based off of Equation 2.2 from arxiv.org/abs/2002.12811.
        
        Parameters
        ----------
            gammasp: float = None, The spike's density index. If not given, defaults to the object's.
            m1: float = None, The mass of the central black hole. If not given, defaults to the object's.
            rhosp: float = None, The density normalization at the spike's size. If not given, defaults to the object's.
            rho6: float = None, The density normalization at 1e-6 [pc]. If not given, defaults to the object's.
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
        """ Converts one normalization constants (rho6/rhosp) [Msun/pc3] to the other.
        
        Parameters
        ----------
            rho6: float = None, The density normalization at 1e-6 [pc]. If not given, defaults to the object's.
            rhosp: float = None, The density normalization at the spike's size. If not given, defaults to the object's.
            gammasp: float = None, The spike's density index. If not given, defaults to the object's.
            m1: float = None, The mass of the central black hole. If not given, defaults to the object's.
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
        """ The initial, total particle density [Msun/pc3] of the power-law profile.
        
        Parameters
        ----------
            r: float, The distance from the center of the spike where the density is evaluated.
        """
        
        return self.rho6 *(r/1e-6)**-self.gammasp # [Msun/pc3]
    
    def feps_init(self, eps: float) -> float:
        """ The initial distribution function of the power-law profile.
        
        Parameters
        ----------
            eps: float = self.eps, The specific energy [m2/s2] of the particles in the dark matter spike.
        """
        r6 = 1e-6 *pc # [m]
        rho6 = self.rho6 *Mo/pc**3 # [kg/m3]
        gammasp = self.gammasp
        m1 = self.binary.m1 # [Msun]
        
        N = rho6 *r6**(gammasp) *gamma(gammasp +1)/gamma(gammasp -1/2) *(G *m1 *Mo)**(-gammasp) /np.sqrt(2 *np.pi)**3
        
        return N *eps**(gammasp -3/2)
    
    def fv_init(self, v: float, r: float) -> float:
        """ The initial 1D-velocity distribution [s/m] of the spike at a given distance [pc].
        
        Parameters
        ----------
            v: float, The velocity magnitude [m/s] of the particle.
            r: float, The distance [pc] from the centre of the spike where the distribution is evaluated.
        """
        Vmax = self.binary.Vmax(r) # [m/s]
        
        Av = 4/np.sqrt(np.pi) *gamma(self.gammasp +1)/gamma(self.gammasp -1/2)
        
        return Av *(1 -v**2/Vmax**2)**(self.gammasp -3/2) *v**2/Vmax**3 # [s/m]
    
    def f_break_static(self, m1: float = None, m2: float = None, gammasp: float = None, rhosp: float = None, rho6: float = None) -> float:
        """ The breaking frequency [Hz] where the gravitational wave energy losses are equal to those of dynamical friction.
        This calculation is based off of Eq. 15 from arxiv.org/pdf/2108.04154.pdf.
        Either rhosp or rho6 needs to be specified.
        
        Parameters
        ----------
            m1: float = None, The mass [Msun] of the seed.
            m2: float = None, The mass [Msun] of the companion.
            gammasp: float = None, The slope index of the density profile.
            rhosp: float = None, The density [Msun/pc3] of the spike at its size.
            rho6: float = None, The density [Msun/pc3] of the spike at 1e-6 [pc].
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
    """ A spike with a PowerLaw density distribution around a binary system with index gammasp, and normalization rho6 [Msun/pc3] at 1e-6 [pc].
    Alternatively to rho6, it can be initialized using the the density rhosp [Msun/pc3] at its size.
    
    This class utilizes analytical expressions based on the power-law distribution function and cannot be evolved, hence 'Static'.
    """
    
    def rho(self, r: float, v_lower: float = 0, v_upper: float = None, chi_lower: float = 0, chi_upper: float = 1) -> float:
        """ The density [Msun/pc3] of the spike at a given distance [pc] from the centre, and within a given velocity range.
        The range can be specified by the ratio of the velocity to the maximum velocity of the spike, chi.
        Alternatively, the range can be specified by the velocity itself.
        
        Parameters
        ----------
            r: float, The distance [pc] from the centre of the spike where the density is evaluated.
            v_lower: float = 0, The upper velocity [m/s] limit.
            v_upper: float = None, The lower velocity [m/s] limit.
            chi_lower: float = 0, The ratio of the upper velocity limit to the maximum velocity of the spike.
            chi_upper: float = 1, The ratio of the lower velocity limit to the maximum velocity of the spike.
        """
        if r <= 0: warn("Distance must be positive. Returning 0."); return 0
        
        Vmax = self.binary.Vmax(r) # [m/s]
        if chi_lower < 0 or v_lower < 0:
            chi_lower = 0; v_lower = 0
            warn("The lower velocity limit should not be negative. Proceed as if zero.")
        
        if chi_upper > 1 or v_lower > Vmax:
            chi_lower = 1; v_lower = Vmax
            warn("The upper velocity limit should not exceed escape velocity. Proceed as if at escape.")
        
        if v_lower > 0:
            if chi_lower > 0: raise ValueError("Only one of v_lower or chi_lower should be given.")
            chi_lower = v_lower/Vmax # [m/s]
        
        if v_upper is not None: 
            if chi_upper < 1: raise ValueError("Only one of v_upper or chi_upper should be given.")
            chi_upper = v_upper/Vmax # [m/s]
        
        if chi_lower == chi_upper: return 0 # If the range has no width, don't do the calculation.
        
        # ----------------------------------------
        xi_DF = self.xi_Nl(0, chi_upper) -self.xi_Nl(0, chi_lower)
        
        return self.rho_init(r) *xi_DF # [Msun/pc3]
    
    @property
    def feps(self) -> float:
        """ The distribution function of the power-law profile, given as its initial value. """
        
        return self.feps_init(self.eps)
    
    def fv(self, v: float, r: float) -> float:
        """ The 1D-velocity distribution [s/m] of the spike at a given distance [pc], given as its initial value.
        
        Parameters
        ----------
            v: float, The velocity magnitude [m/s] of the particle.
            r: float, The distance [pc] from the centre of the spike where the distribution is evaluated.
        """
        
        return self.fv_init(v, r)
    
    def xi_Nl(self, N: float, chi: float) -> float:
        """ The N-th velocity moment normalized by a cutoff velocity $u = chi V_{max}$, where $V_{max}$ is the escape velocity,
        and for particles moving slower than that.
        For example for N = 1, it is the average of $v/(u *V_{max})$.
        
        Parameters
        ----------
            N: float, The index of the velocity moment.
            chi: float, The ratio between the cutoff velocity and the escape velocity of the medium.
        """
        
        A_v = 4/np.sqrt(np.pi) *gamma(self.gammasp +1)/gamma(self.gammasp -1/2)
        
        return A_v *chi**3 / (N +3) *hyp2f1(3/2 -self.gammasp, (N +3)/2, (N +5)/2, chi**2)

    def xi_Nu(self, N: float, chi: float) -> float:
        """ The N-th velocity moment normalized by a cutoff velocity $u = chi V_{max}$, where $V_{max}$ is the escape velocity,
        and for particles moving faster than that.
        For example for N = 1, it is the average of $v/(u *V_{max})$.
        
        Parameters
        ----------
            N: float, The index of the velocity moment.
            chi: float, The ratio between the cutoff velocity and the escape velocity of the medium.
        """
        
        return self.xi_Nl(N, 1) - self.xi_Nl(N, chi)