from spikit.units import *
from numpy import sqrt, cos, sin, linspace

import matplotlib.pyplot as plt

# ============================
# ======== Black Hole ========
class BlackHole:
    """ A black hole with mass m [Msun]. """
    
    def __init__(self, m: float):
        self.m = m # [Msun]
    
    @property
    def Rm(self) -> float:
        """ The mass length-scale [pc] of the black hole. """
        
        return G *(self.m *Mo) /c**2 /pc # [pc]
    
    @property
    def Rs(self) -> float:
        """ The Swartzchild radius [pc] of the black hole. """
        
        return 2 *self.Rm # [pc]
    
    @property
    def Risco(self) -> float:
        """ The innermost stable circular orbit (ISCO) radius [pc] of the black hole. """
        
        return 6 *self.Rm # [pc]
    
    def V(self, r: float) -> float: # Warning, changing the default function will NOT accurately reflect in the code.
        """ The gravitational potential [J/kg] of the black hole at a given distance [pc] from it.
        
        Parameters
        ----------
            r: float, The distance [pc] from the centre of the black hole where the potential is evaluated.
        """
        
        return -G *self.m *Mo /(r *pc) # [J/kg]
    
    def Psi(self, r: float) -> float:
        """ The relative (with respect to infinity) gravitational potential [J/kg] of the black hole at a given distance [pc] from it.
        
        Parameters
        ----------
            r: float, The distance [pc] from the centre of the black hole where the potential is evaluated.
        """
        V_infty = 0 # Taken to be zero.
        
        return V_infty -self.V(r) # [J/kg]

# ========================
# ======== Binary ========
class Binary:
    """ A binary composed of a large, central black hole with mass m1 [Msun], and an orbiting companion with m2 [Msun]. """
    
    def __init__(self, m1: float, m2: float):
        
        if m1 < m2: raise ValueError("m1 must be greater than m2.")
        
        self.m1 = m1 # [Msun] Central black hole.
        self.m2 = m2 # [Msun] Companion.
    
    @property
    def M1(self) -> BlackHole:
        """ A BlackHole object that represents the central black hole. """
        return BlackHole(self.m1)
    
    @property
    def M2(self) -> BlackHole:
        """ A BlackHole object that represents the orbiting companion. """
        return BlackHole(self.m2)
    
    def m(self, m1: float = None, m2: float = None) -> float:
        """ The total mass [Msun] of the binary.
        
        Parameters
        ----------
            m1: float = self.m1, The mass [Msun] of the central black hole.
            m2: float = self.m2, The mass [Msun] of the secondary black hole.
        """
        if m1 is None: m1 = self.m1
        if m2 is None: m2 = self.m2
        
        return m1 +m2 # [Msun]
    
    def mu(self, m1: float = None, m2: float = None) -> float:
        """ The reduced mass [Msun] of the binary.
        
        Parameters
        ----------
            m1: float = self.m1, The mass [Msun] of the central black hole.
            m2: float = self.m2, The mass [Msun] of the secondary black hole.
        """
        if m1 is None: m1 = self.m1
        if m2 is None: m2 = self.m2
        
        return m1 *m2/(m1 +m2) # [Msun]
    
    def Mchirp(self, m1: float = None, m2: float = None) -> float:
        """ The chirp mass [Msun] of the binary.
        
        Parameters
        ----------
            m1: float = self.m1, The mass [Msun] of the central black hole.
            m2: float = self.m2, The mass [Msun] of the secondary black hole.
        """
        if m1 is None: m1 = self.m1
        if m2 is None: m2 = self.m2
        
        return (m1 *m2)**(3/5) / (m1 +m2)**(1/5) # [Msun]
    
    def q(self, m1: float = None, m2: float = None) -> float:
        """ The mass ratio [Msun] of the binary.
        
        Parameters
        ----------
            m1: float = self.m1, The mass [Msun] of the central black hole.
            m2: float = self.m2, The mass [Msun] of the secondary black hole.
        """
        if m1 is None: m1 = self.m1
        if m2 is None: m2 = self.m2
        
        return m2/m1
    
    def Risco(self, m: float = None) -> float:
        """ The Innermost Stable Circular Orbit (ISCO) [pc] of a black hole in the spike.
        This is a convenience function, which by default takes the central black hole of the binary.
        Solvers use this value as a reference to approximate the merger.
        
        Parameters
        ----------
            m: float = self.m1, The mass [Msun] of the central black hole.
        """
        if m is None: m = self.m1 # [Msun]
        
        return BlackHole(m).Risco # [pc]
    
    def rhill(self, r2: float, q: float = None) -> float:
        """ The Hill radius [pc] of the binary.
        This is a scale measurement of the companion's relative gravitational influence and
        is used as an estimate to the maximum impact parameter which influences particles.
        
        Parameters
        ----------
            r2: float, The binary's separation [pc] where the Hill radius is evaluated at.
            q: float = self.q(), The mass ratio m2/m1 of the binary.
        """
        if q is None: q = self.q()
        
        return r2 *(q/3)**(1/3) # [pc]
    
    # ======== Orbital Properties ========
    
    def p(self, a: float, e: float = 0) -> float:
        """ The semi-latus rectum [units] of an elliptic orbit.
        
        Parameters
        ----------
            a: float, The semi-major axis [units] of the orbit.
            e: float = 0, The eccentricity of the orbit.
        """
        
        return a *(1 -e**2) # [units]
    
    def r2(self, a: float = None, e = 0, theta: float = 0) -> float:
        """ The binary's separation [units] at a given true anomaly [rad] of an elliptic inspiral.
        
        Parameters
        ----------
            a: float, The semi-major axis [units] of the orbit.
            e: float = 0, The eccentricity of the orbit.
            theta: float = 0, The true anomaly [rad] corresponding to the binary's position.
                This angle is measured such that it is 0 or 2pi at the orbit's periapsis, and pi at its apoapsis.
        """
        
        return self.p(a, e) /(1 +e *cos(theta)) # [units]
    
    def rperi(self, a: float = None, e = 0) -> float:
        """ The binary's periapsis [units].
        This is the minimum separation available to the two components, evaluated at a true anomaly of 0.
        
        Parameters
        ----------
            a: float, The semi-major axis [units] of the orbit.
            e: float = 0, The eccentricity of the orbit.
        """
        return self.r2(a, e, 0) # [units]
    
    def rapo(self, a: float = None, e = 0) -> float:
        """ The binary's apoapsis [units].
        This is the maximum separation available to the two components, evaluated at a true anomaly of pi.
        
        Parameters
        ----------
            a: float, The semi-major axis [units] of the orbit.
            e: float = 0, The eccentricity of the orbit.
        """
        return self.r2(a, e, pi) # [units]
    
    def Vmax(self, r: float, m1: float = None) -> float:
        """ The maximum orbital velocity [m/s] available to objects that orbit a black hole.
        This represents the escape velocity for a system, and by default accounts for the largest black hole's gravity.
        
        Parameters
        ----------
            r: float, The distance at which the escape velocity [m/s] is evaluated at.
            m1: float = self.m1, The mass [Msun] of the black hole in question.
        """
        if m1 is None: m1 = self.m1 # [Msun]
        
        return sqrt(2 *G *m1 *Mo /(r *pc)) # [m/s]
    
    def u(self, r2: float, a: float = None, m: float = None) -> float:
        """ The orbital velocity [m/s] of a a reduced object in an elliptic binary at its current separaton.
        If a is not specified, a circular orbit is assumed.
        
        Parameters
        ----------
            r2: float, The separation [pc] of the binary where the orbital velocity is evaluated at.
            a: float = r2, The semi-major axis [pc] of the orbit.
            m: float = self.m(), The total mass [Msun] of the black holes.
        """
        
        if a is None: a = r2 # [pc]
        if m is None: m = self.m() # [Msun]
        
        return sqrt(G *self.m() *Mo *(2/r2 -1/a)/pc) # [m/s]
    
    def T(self, a: float, m: float = None) -> float:
        """ The orbital period [s] of an elliptic binary.
        
        Parameters
        ----------
            a: float, The semi-major axis [pc] of the orbit.
            m: float = self.m(), The total mass [Msun] of the black holes.
        """
        if m is None: m = self.m() # [Msun]
        
        return 2 *pi *sqrt((a *pc)**3 /(G *m *Mo)) # [s]
    
    def f(self, a: float, m: float = None) -> float:
        """ The orbital frequency [Hz] of an elliptic binary.
        
        Parameters
        ----------
            a: float, The semi-major axis [pc] of the orbit.
            m: float = self.m(), The total mass [Msun] of the black holes.
        """
        
        if m is None: m = self.m() # [Msun]
        
        return 1/self.T(a, m) # [Hz]
    
    def w(self, a: float, m: float = None) -> float:
        """ The orbital angular velocity [Hz] of an elliptic binary.
        
        Parameters
        ----------
            a: float, The semi-major axis [pc] of the orbit.
            m: float = self.m(), The total mass [Msun] of the black holes.
        """
        
        if m is None: m = self.m() # [Msun]
        
        return 2*pi *self.f(a, m) # [Hz]
    
    def u1(self, r2: float, a: float = None, m1: float = None, m2: float = None) -> float:
        """ The orbital velocity [m/s] of the central object.
        If a is not specified, a circular orbit is assumed.
        
        Parameters
        ----------
            r2: float, The separation [pc] of the binary where the orbital velocity is evaluated at.
            a: float = r2, The semi-major axis [pc] of the orbit.
            m1: float = self.m1, The mass [Msun] of the largest black hole.
            m2: float = self.m2, The mass [Msun] of the smallest black hole.
        """
        
        if m1 is None: m1 = self.m1 # [Msun]
        if m2 is None: m2 = self.m2 # [Msun]
        if a is None: a = r2 # [pc]
        
        return self.u(r2, a) *m2/(m1 +m2) # [m/s]
    
    def u2(self, r2: float, a: float = None, m1: float = None, m2: float = None) -> float:
        """ The orbital velocity [m/s] of the central object.
        If a is not specified, a circular orbit is assumed.
        
        Parameters
        ----------
            r2: float, The separation [pc] of the binary where the orbital velocity is evaluated at.
            a: float = r2, The semi-major axis [pc] of the orbit.
            m1: float = self.m1, The mass [Msun] of the largest black hole.
            m2: float = self.m2, The mass [Msun] of the smallest black hole.
        """
        
        if m1 is None: m1 = self.m1 # [Msun]
        if m2 is None: m2 = self.m2 # [Msun]
        if a is None: a = r2 # [pc]
        
        return self.u(r2, a) *m1/(m1 +m2) # [m/s]
    
    # ======== Conservative Quantities ========
    
    def Eorb(self, a: float, m1: float = None, m2: float = None) -> float:
        """ The orbital energy [J] that describes the binary.
        This is a negative number, therefore orbits will tend to shrink when it decreases.
        
        Parameters
        ----------
            a: float, The semi-major axis [pc] of the orbit.
            m1: float = self.m1, The mass [Msun] of the largest black hole.
            m2: float = self.m2, The mass [Msun] of the smallest black hole.
        """
        
        if m1 is None: m1 = self.m1 # [Msun]
        if m2 is None: m2 = self.m2 # [Msun]
        
        return -G *m1 *m2 *Mo**2/(2 *a *pc) # [J]
    
    def Lorb(self, a: float, e: float = 0, m1: float = None, m2: float = None) -> float:
        """ The orbital angular moment [Js] that describes the binary.
        
        Parameters
        ----------
            a: float, The semi-major axis [pc] of the orbit.
            e: float = 0, The eccentricity of the elliptic orbit.
            m1: float = self.m1, The mass [Msun] of the largest black hole.
            m2: float = self.m2, The mass [Msun] of the smallest black hole.
        """
        
        if m1 is None: m1 = self.m1 # [Msun]
        if m2 is None: m2 = self.m2 # [Msun]
        
        m = self.m(m1, m2) # [Msun]
        mu = self.mu(m1, m2) # [Msun]
        
        return mu *Mo *sqrt(G *m *Mo *self.p(a, e) *pc)
    
    def a(self, Eorb: float, m1: float = None, m2: float = None) -> float:
        """ The semi-major axis [pc] of an elliptic binary.
        
        Parameters
        ----------
            Eorb: float, The orbital energy of the binary when the semi-major axis [pc] is evaluated at.
            m1: float = self.m1, The mass [Msun] of the largest black hole.
            m2: float = self.m2, The mass [Msun] of the smallest black hole.
        """
        
        if m1 is None: m1 = self.m1 # [Msun]
        if m2 is None: m2 = self.m2 # [Msun]
        
        return -G *m1 *m2 *Mo**2/(2 *Eorb) /pc # [pc]
    
    def e(self, Eorb: float, Lorb: float, m1: float = None, m2: float = None) -> float:
        """ The eccentricity of an elliptic binary.
        
        Parameters
        ----------
            Eorb: float, The orbital energy of the binary when the eccentricity is evaluated at.
            Lorb: float, The orbital angular momentum of the binary when the eccentricity is evaluated at.
            m1: float = self.m1, The mass [Msun] of the largest black hole.
            m2: float = self.m2, The mass [Msun] of the smallest black hole.
        """
        
        if m1 is None: m1 = self.m1 # [Msun]
        if m2 is None: m2 = self.m2 # [Msun]
        
        mu = self.mu(m1, m2) # [Msun]
        m = self.m(m1, m2) # [Msun]
        
        a = self.a(Eorb) # [pc] 
        p = (Lorb/mu/Mo)**2 / (G *m *Mo) /pc # [pc]
        
        return sqrt(1 -p/a)
    
    # ======== Evolution ========
    
    def da_dt(self, dE_dt: float, dm2_dt: float, r2: float, a: float, m: float = None) -> float:
        """ The rate of change of the semi-major axis [pc/s] due to energy and mass change.
        
        Parameters
        ----------
            dE_dt: float, The rate of change for the binary's orbital energy [J/s]. Losses should be negative.
            dm2_dt: float, The rate of change for the companion's mass [Msun/s]. Positive for increasing mass.
            r2: float, The separation [pc] of the binary where the orbital velocity is evaluated at.
            a: float, The semi-major axis [pc] of the orbit.
            m: float = self.m(), The total mass [Msun] of the binary.
        """
        
        if m is None: m = self.m() # [Msun]
        
        return -a *( dE_dt/self.Eorb(a) + dm2_dt/m *(2 *a/r2 -1)) # [pc/s]
    
    def de_dt(self, dE_dt: float, dL_dt: float, dm2_dt: float, r2: float, a: float, e: float, m1: float = None, m2: float = None) -> float:
        """ The rate of change of the eccentricity [1/s] due to energy, angular momentum and mass change.
        
        Parameters
        ----------
            dE_dt: float, The rate of change for the binary's orbital energy [J/s]. Losses should be negative.
            dL_dt: float, The rate of change for the binary's orbital angular momentum [J]. Losses should be negative.
            dm2_dt: float, The rate of change for the companion's mass [Msun/s]. Positive for increasing mass.
            r2: float, The separation [pc] of the binary where the orbital velocity is evaluated at.
            a: float, The semi-major axis [pc] of the orbit.
            m1: float = self.m1, The mass [Msun] of the largest black hole.
            m2: float = self.m2, The mass [Msun] of the smallest black hole.
        """
        
        if e == 0: return 0
        if m1 is None: m1 = self.m1 # [Msun]
        if m2 is None: m2 = self.m2 # [Msun]
        
        m = m1 +m2 # [Msun]
        return - (1 -e**2)/e *( dE_dt/2/self.Eorb(a) + (m1/m) *dL_dt/self.Lorb(a, e) + dm2_dt/m *(a/r2 -1)) # [1/s]
    
    # ======== Misc ========
    
    def draw_orbit(self, a: float, e: float = 0, m1: float = None, m2: float = None):
        """ Draws the stationary orbit of two black holes.
        
        Parameters
        ----------
            a: float, The semi-major axis [pc] of the orbit.
            e: float = 0, The eccentricity.
            m1: float = self.m1, The mass [Msun] of the largest black hole.
            m2: float = self.m2, The mass [Msun] of the smallest black hole.
        """
        
        if e < 0 or e >= 1: raise ValueError("The eccentricity must be between 0 and 1.")
        
        if m1 is None: m1 = self.m1 # [Msun]
        if m2 is None: m2 = self.m2 # [Msun]
        
        theta = linspace(0, 2 *pi, 1000)

        M1 = BlackHole(m1)
        
        r2 = self.r2(a, e, theta)
        x = r2 *cos(theta)
        y = r2 *sin(theta)

        plt.figure(figsize = (5, 4.5))
        plt.plot(x, y, linewidth = 1)
        plt.axis('equal')
        plt.xlabel("Separation x [pc]")
        plt.ylabel("Separation y [pc]")

        x = M1.Rs *cos(theta)
        y = M1.Rs *sin(theta)
        plt.fill_between(x, y, color = "black")