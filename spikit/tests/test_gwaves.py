from spikit.forces import GravitationalWaves
from spikit.binary import Binary

from spikit.tests.fixtures import default_binary

def test_no_mass_change_from_gw(default_binary: Binary):
    assert GravitationalWaves(default_binary).dm2_dt(1, 1) == 0 # [N]