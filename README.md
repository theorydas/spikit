<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) -->
# Spikit (Spike-Kit)

### `Exploring the dark universe through gravitational waves`

This package aims to assist in the search for the nature of dark matter through the lens of gravitational wave astronomy. It is a collection of user-friendly tools for simulating the inspiral of black hole binaries enveloped by `dark matter spikes'.

# Getting started

You should first make sure to have installed the dependencies listed in the setup.py file. Then, you can install the package by running the following command in the root directory of the package:

`pip install .` or `pip install -e .` (for development).

## Usage

The package is designed around the idea of black hole binaries inside dark matter environments. The main classes are the `Binary` class, that describes the black holes and their motion, and `Spike`, an abstract class that describes the environment.

Additionally, there are `Force` and `Feedback` classes that control the interactions between the binary partners or with the environment. The latter is used to update the distribution function of the environment, while the former is calculated by that distribution function. A `Solver` class is used to evolve the binary in time. For example:

```python
binary = Binary(m1 = 1e4, m2 = 10)
spike = StaticPowerLaw(binary, gammasp = 7/3, rho6 = 5.448e15)

# BH-BH interaction.
gw = GravitationalWaves(binary)
# BH-Spike interactions.
df = DynamicalFrictionIso(spike)
acc = AccretionIso(spike)

results_gw = DynamicSolver(binary, loss = [gw, df, acc]).solve(a0)
```

## Blueprints

The package offers a set of blueprints that can be used to quickly set up a simulation, or utilize analytical solutions.

