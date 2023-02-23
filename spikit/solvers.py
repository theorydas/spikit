from spikit.binary import Binary
from spikit.forces import GravitationalWaves, Force
from spikit.spike import Spike, StaticPowerLaw

from typing import Union
import numpy as np

class DynamicSolver():
    """ The solver class that utilizes a dynamic time step to solve the binary evolution problem."""
    
    def __init__(self, binary: Binary = None, spike: Spike = None, loss: Union[GravitationalWaves, Force, list] = [], feedback: Union[GravitationalWaves, Force, list] = []) -> None:
        
        # A list of all the forces and feedback mechanisms that are active.
        self.loss = loss if type(loss) is list else [loss]
        self.feedback = feedback if type(feedback) is list else [feedback]
        
        if len(self.loss) == 0:
            raise ValueError("At least one loss mechanism must be given.")
        
        if spike == None and len(self.feedback) > 0:
            raise ValueError("Feedback mechanisms are only supported for spike models.")
        
        if (binary is not None and spike is not None) or (binary is None and spike is None):
            raise ValueError("Exactly one of binary or spike must be given.")
        
        self.binary = binary or spike.binary
        self.spike = spike or StaticPowerLaw.from_binary(binary, 7/3, 1) # If no spike is given, use a near zero-density power law spike as placeholder.
    
    def _integrate_order_2(self, parameters: dict, h: float) -> tuple:
        """ A modified, second order method for solving the static problem."""
        a = parameters["a"][-1]; t = parameters["t"][-1]; m2 = parameters["m2"][-1]
        
        # First step
        r2 = a; u = self.binary.u2(r2, m1 = self.binary.m1, m2 = m2)
        
        dm2dt_1 =  -sum([loss.dm2_dt(r2, u) for loss in self.losses])
        dEdt_1 = -sum([loss.dE_dt(r2, u) for loss in self.losses])
        dadt_1 = self.binary.da_dt(dE_dt = dEdt_1, dm2_dt = dm2dt_1, r2 = r2, a = a, m = self.binary.m1 +m2) # [pc/s]
        
        dt = abs(a/dadt_1) *h # [s]

        a += 2/3 *dadt_1 *dt # [pc]
        m2 += 2/3 *dm2dt_1 *dt # [pc]
        
        # Second step
        r2 = a; u = self.binary.u2(r2, m1 = self.binary.m1, m2 = m2)
        
        dm2dt_2 = sum([loss.dm2_dt(r2, u) for loss in self.losses])
        dEdt_2 = -sum([loss.dE_dt(r2, u) for loss in self.losses])
        dadt_2 = self.binary.da_dt(dE_dt = dEdt_2, dm2_dt = dm2dt_1, r2 = r2, a = a, m = self.binary.m1 +m2) # [pc/s]
        
        t += dt # [s]
        a += dt/12 *(9 *dadt_2 -5 *dadt_1) # [pc]
        m2 += dt/12 *(9 *dm2dt_2 -5 *dm2dt_1) # [Mo]
        
        new_values = {"t": [t], "a": [a], "m2": [self.binary.m2]}
        # Update the parameters dictionary to append each new value.
        for key in parameters.keys():
            parameters[key] += new_values[key]
        
        return parameters
    
    def _integrate_order_1(self, parameters: dict, h: float) -> tuple:
        """ A 1st order solver."""
        a = parameters["a"][-1]; t = parameters["t"][-1]; m2 = parameters["m2"][-1]
        
        r2 = a; u = self.binary.u2(r2, m1 = self.binary.m1, m2 = m2)
        
        dm2dt = sum([loss.dm2_dt(r2, u) for loss in self.losses])
        dEdt = -sum([loss.dE_dt(r2, u) for loss in self.losses])
        dadt = self.binary.da_dt(dE_dt = dEdt, dm2_dt = dm2dt, r2 = r2, a = a) # [pc/s]
        
        dt = abs(a/dadt) *h # [s]

        a += dadt *dt # [pc]
        m2 += dm2dt *dt # [Mo]
        t += dt # [s]
        
        return a, m2, t
    
        new_values = {"t": [t], "a": [a], "m2": [self.binary.m2]}
        # Update the parameters dictionary to append each new value.
        for key in parameters.keys():
            parameters[key] += new_values[key]
        
        return parameters
    
    def solve(self, a0: float, e0: float = 0, h: float = 1e-2, order: int = 2) -> tuple:
        """
        Solve the static problem. 
        a0 is the initial semi-major axis [pc], e0 the initial eccentricity,
        h is the (maximum) relative change of the non-time quantities per step.
        """
        # Raise an error if the initial conditions are not valid.
        if e0 < 0 or e0 >= 1: raise ValueError("The eccentricity must be between 0 and 1.")
        if a0 *(1 -e0) <= self.binary.Risco(): raise ValueError("The initial periapsis must be larger than the Innermost Stable Circular Orbit.")
        if order not in [1, 2]: raise ValueError("The order must be 1 or 2.")
            
        # Setup binary and gravitational wave losses.
        risco = self.binary.Risco()
        integrator = self._integrate_order_1 if order == 1 else self._integrate_order_2
        
        # Evolve the binary.
        parameters = {"t": [0], "a": [a0], "m2": [self.binary.m2]}
        
        while parameters["a"][-1] > risco:         
            # Integrate the system by updating the previous dictioanry.
            parameters = integrator(parameters, h = h)
        
        # Must interpolate the last entry of each array to the risco.
        for key in parameters.keys():
            # Convert the arrays are numpy arrays.
            parameters[key] = np.array(parameters[key])
            
            if key == "a":
                # Simply replace the last value with risco.
                parameters[key][-1] = risco
                continue
            
            # Iterpolate the last step to the risco.
            approximate_last = parameters[key][-1]
            rate = (approximate_last -parameters[key][-2]) /(parameters["a"][-1] -parameters["a"][-2])
            interpolated_last = (risco -parameters["a"][-1]) * rate +approximate_last
            
            # Update the last value.
            parameters[key][-1] = interpolated_last
        
        return parameters