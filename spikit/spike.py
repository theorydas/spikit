from spikit.binary import black_hole
from scipy.special import gamma, hyp2f1
import numpy as np

class spike:
    """ The default spike class. """
    def __init__(self) -> None:
        pass

class static_powerlaw(spike):
    """ An isotropic, power-law spike with a power-law index gammasp. """
    
    def __init__(self, seed: black_hole, gammasp: float, rho6: float = 0):
        self.gammasp = gammasp # Power-law index of the spike.
        self.rho6 = rho6
        self.seed = seed
        
        #  The normalization of the power law distribution function.
        self.A_f = 4/np.sqrt(np.pi) *gamma(self.gammasp +1)/gamma(self.gammasp -1/2)

    def rho(self, r: float) -> float:
        return self.rho6 *(r/1e-6)**-self.gammasp
    
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
        
        return self.A_f / (N +3) *( chi**-N * hyp2f1(3/2 -self.gammasp, (N +3)/2, (N +5)/2, 1) -chi**3 *hyp2f1(3/2 -self.gammasp, (N +3)/2, (N +5)/2, chi**2) )
