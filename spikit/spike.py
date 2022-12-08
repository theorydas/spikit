from scipy.special import gamma, hyp2f1
import numpy as np

class static_spike:
    """ An isotropic, power-law spike with a power-law index gammasp. """
    
    def __init__(self, seed: black_hole, gammasp: float, rho6: float = 0):
        self.gamma = gammasp # Power-law index of the spike.
        self.rho6 = rho6
        self.seed = seed
    
    @property
    def A_f(self) -> float:
        """ The normalization of the power law distribution function."""
        A = 4/np.sqrt(np.pi) *gamma(self.gammasp +1)/gamma(self.gammasp -1/2)
        
        return A

    def ksi_kl(self, k: float, chi: float) -> float:
        """ Returns the lower, normalized velocity moment (v/u)^k for particles with v < u.
        
            chi is the ratio of the orbital velocity to the escape velocity.
            k is the order of the velocity-moment.
        """
        
        return self.A_f *chi**3 / (k +3) *hyp2f1(3/2 -self.gammasp, (k +3)/2, (k +5)/2, chi**2)

    def ksi_ku(self, k: float, chi: float) -> float:
        """ Returns the upper, normalized velocity moment (v/u)^k for particles with v > u.
        
            chi is the ratio of the orbital velocity to the escape velocity.
            k is the order of the velocity-moment.
        """
        
        return self.A_f / (k +3) *( chi**-k * hyp2f1(3/2 -self.gammasp, (k +3)/2, (k +5)/2, 1) -chi**3 *hyp2f1(3/2 -self.gammasp, (k +3)/2, (k +5)/2, chi**2) )
