from spikit.binary import Binary
from spikit.forces import GravitationalWaves, Force

from typing import Union
import numpy as np

class StaticSolver():
    """ The solver class for the static problem. """
    
    def __init__(self, binary: Binary, losses: Union[GravitationalWaves, Force, list]) -> None:
        # A list of all the forces that are active.
        self.losses = losses if type(losses) is list else [losses] 
        self.binary = binary
        
        pass
    
    def _integrate_order_2(self, a: float, t: float, h: float) -> tuple: # TODO: Implement mass change also.
        """ A modified, second order method for solving the static problem."""
        # First step
        r2 = a; u = self.binary.u2(r2)
        m = self.binary.m1 +self.binary.m2
        
        dEdt_1 = -sum([loss.dE_dt(r2, u) for loss in self.losses])
        dadt_1 = self.binary.da_dt(dE_dt = dEdt_1, dm2_dt = 0, r2 = r2, a = a, m = m) # [pc/s]

        dt = abs(a/dadt_1) *h # [s]

        a += 2/3 *dadt_1 *dt # [pc]
        # Second step
        r2 = a; u = self.binary.u2(r2)
        
        dEdt_2 = -sum([loss.dE_dt(r2, u) for loss in self.losses])
        dadt_2 = self.binary.da_dt(dE_dt = dEdt_2, dm2_dt = 0, r2 = r2, a = a, m = m) # [pc/s]
        
        t += dt # [s]
        a += dt/12 *(9 *dadt_2 -5 *dadt_1) # [pc]
        
        return a, t
    
    def _integrate_order_1(self, a: float, t: float, h: float) -> tuple:
        """ A 1st order solver."""
        r2 = a; u = self.binary.u2(r2)
        m = self.binary.m1 +self.binary.m2
        
        dEdt = -sum([loss.dE_dt(r2, u) for loss in self.losses])
        dadt = self.binary.da_dt(dE_dt = dEdt, dm2_dt = 0, r2 = r2, a = a, m = m) # [pc/s]

        dt = abs(a/dadt) *h # [s]

        a += dadt *dt # [pc]
        t += dt # [s]
        
        return a, t
    
    def solve(self, a0: float, e0: float = 0, h: float = 1e-2, order: int = 2):
        """
        Solve the static problem. 
        a0 is the initial semi-major axis [pc], e0 the initial eccentricity,
        h is the (maximum) relative change of the non-time quantities per step.
        """
        # Raise an error if the initial conditions are not valid.
        if e0 < 0 or e0 >= 1: raise ValueError("The eccentricity must be between 0 and 1.")
        if a0 <= self.binary.Risco(): raise ValueError("The initial separation must be larger than the innermost stable circular orbit.")
        if order not in [1, 2]: raise ValueError("The order must be 1 or 2.")
        
        # Setup binary and gravitational wave losses.
        risco = self.binary.Risco()
        integrator = self._integrate_order_1 if order == 1 else self._integrate_order_2
        
        # Evolve the binary.
        a_list = [a0]
        t_list = [0]

        while a_list[-1] > risco:
            a = a_list[-1]; t = t_list[-1]
            
            a_, t_ = integrator(a, t, h = h)
            a_list.append(a_); t_list.append(t_)

        t = np.array(t_list)
        a = np.array(a_list)

        # Interpolate the last step to the risco.
        t_final = (risco -a[-1]) *(t[-2] -t[-1])/(a[-2] -a[-1]) +t[-1]

        t = np.append(t[:-1], t_final)
        a = np.append(a[:-1], risco)
        
        return t, a