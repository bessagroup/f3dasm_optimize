#                                                                       Modules
# =============================================================================

# Standard
from typing import Optional

# Third-party
import pygmo as pg

# Locals
from ._protocol import OptimizerTuple
from .adapters.pygmo_implementations import PygmoAlgorithm

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


def cmaes_pygmo(population: int = 30,
                force_bounds: bool = True,
                seed: Optional[int] = None, **kwargs) -> OptimizerTuple:
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
    Adapted from pygmo.

    Parameters
    ----------
    population : int, optional
        The population size, by default 30
    force_bounds : bool, optional
        If bounds should be enforced, by default True
    seed : Optional[int], optional
        Seed for random number generation, by default None

    Returns
    -------
    OptimizerTuple
        A configured instance of the CMA-ES optimizer.
    """
    return OptimizerTuple(
        base_class=PygmoAlgorithm,
        algorithm=pg.cmaes,
        hyperparameters={
            'population': population,
            'force_bounds': force_bounds,
            'seed': seed,
            'memory': True,
            'gen': 1,
            **kwargs
        })

# =============================================================================


def de_pygmo(population: int = 30,
             F: float = 0.8,
             CR: float = 0.9,
             variant: int = 2,
             ftol: float = 0.0,
             xtol: float = 0.0,
             seed: Optional[int] = None, **kwargs) -> OptimizerTuple:
    """
    Differential Evolution optimizer using pygmo.

    Parameters
    ----------
    population : int, optional
        The population size, by default 30
    F : float, optional
        Scaling factor, by default 0.8
    CR : float, optional
        Crossover rate, by default 0.9
    variant : int, optional
        Variant to use, by default 2
    ftol : float, optional
        Function tolerance, by default 0.0
    xtol : float, optional
        Variable tolerance, by default 0.0
    seed : Optional[int], optional
        Seed for random number generation, by default None

    Returns
    -------
    PygmoAlgorithm
        A configured instance of the Differential Evolution optimizer.
    """
    return OptimizerTuple(
        base_class=PygmoAlgorithm,
        algorithm=pg.de,
        hyperparameters={
            'population': population,
            'F': F,
            'CR': CR,
            'variant': variant,
            'ftol': ftol,
            'xtol': xtol,
            'seed': seed,
            'gen': 1,
            **kwargs
        })


# =============================================================================

def pso_pygmo(population: int = 30,
              eta1: float = 2.05,
              eta2: float = 2.05,
              seed: Optional[int] = None, **kwargs) -> OptimizerTuple:
    """
    Particle Swarm Optimization (Generational) using pygmo.

    Parameters
    ----------
    population : int, optional
        The population size, by default 30
    eta1 : float, optional
        Cognitive acceleration coefficient, by default 2.05
    eta2 : float, optional
        Social acceleration coefficient, by default 2.05
    seed : Optional[int], optional
        Seed for random number generation, by default None

    Returns
    -------
    PygmoAlgorithm
        A configured instance of the PSO optimizer.
    """
    return OptimizerTuple(
        base_class=PygmoAlgorithm,
        algorithm=pg.pso_gen,
        hyperparameters={
            'population': population,
            'eta1': eta1,
            'eta2': eta2,
            'seed': seed,
            'memory': True,
            'gen': 1,
            **kwargs
        })


# =============================================================================


def sade(population: int = 30,
         variant: int = 2,
         variant_adptv: int = 1,
         ftol: float = 0.0,
         xtol: float = 0.0,
         seed: Optional[int] = None, **kwargs) -> OptimizerTuple:
    """
    Self-adaptive Differential Evolution (SADE) using pygmo.

    Parameters
    ----------
    population : int, optional
        The population size, by default 30
    variant : int, optional
        Variant to use, by default 2
    variant_adptv : int, optional
        Adaptive variant type, by default 1
    ftol : float, optional
        Function tolerance, by default 0.0
    xtol : float, optional
        Variable tolerance, by default 0.0
    seed : Optional[int], optional
        Seed for random number generation, by default None

    Returns
    -------
    PygmoAlgorithm
        A configured instance of the SADE optimizer.
    """
    return OptimizerTuple(
        base_class=PygmoAlgorithm,
        algorithm=pg.sade,
        hyperparameters={
            'population': population,
            'variant': variant,
            'variant_adptv': variant_adptv,
            'ftol': ftol,
            'xtol': xtol,
            'seed': seed,
            'memory': True,
            'gen': 1,
            **kwargs
        })


# =============================================================================

def sea(population: int = 30,
        seed: Optional[int] = None, **kwargs) -> OptimizerTuple:
    """
    Simple Evolutionary Algorithm (SEA) optimizer using pygmo.

    Parameters
    ----------
    population : int, optional
        The population size, by default 30
    seed : Optional[int], optional
        Seed for random number generation, by default None

    Returns
    -------
    PygmoAlgorithm
        A configured instance of the SEA optimizer.
    """
    return OptimizerTuple(
        base_class=PygmoAlgorithm,
        algorithm=pg.sea,
        hyperparameters={
            'population': population,
            'seed': seed,
            'gen': 1,
            **kwargs
        })


# =============================================================================

def sga(population: int = 30,
        cr: float = 0.9,
        eta_c: float = 1.0,
        m: float = 0.02,
        param_m: float = 1.0,
        param_s: int = 2,
        crossover: str = 'exponential',
        mutation: str = 'polynomial',
        selection: str = 'tournament',
        seed: Optional[int] = None,
        **kwargs) -> OptimizerTuple:
    """
    Simple Genetic Algorithm (SGA) optimizer using pygmo.

    Parameters
    ----------
    population : int, optional
        The population size, by default 30
    cr : float, optional
        Crossover probability, by default 0.9
    eta_c : float, optional
        Crossover distribution index, by default 1.0
    m : float, optional
        Mutation probability, by default 0.02
    param_m : float, optional
        Mutation distribution index, by default 1.0
    param_s : int, optional
        Selection pressure, by default 2
    crossover : str, optional
        Crossover method, by default 'exponential'
    mutation : str, optional
        Mutation method, by default 'polynomial'
    selection : str, optional
        Selection method, by default 'tournament'
    seed : Optional[int], optional
        Seed for random number generation, by default None

    Returns
    -------
    PygmoAlgorithm
        A configured instance of the SGA optimizer.
    """
    return OptimizerTuple(
        base_class=PygmoAlgorithm,
        algorithm=pg.sga,
        hyperparameters={
            'population': population,
            'cr': cr,
            'eta_c': eta_c,
            'm': m,
            'param_m': param_m,
            'param_s': param_s,
            'crossover': crossover,
            'mutation': mutation,
            'selection': selection,
            'seed': seed,
            'gen': 1,
            **kwargs
        })


# =============================================================================

def simanneal_pygmo(population: int = 30,
                    Ts: float = 10.0,
                    Tf: float = 0.1,
                    n_T_adj: int = 10,
                    n_range_adj: int = 10,
                    bin_size: int = 10,
                    start_range: float = 1.0,
                    seed: Optional[int] = None, **kwargs) -> OptimizerTuple:
    """
    Simulated Annealing optimizer using pygmo.

    Parameters
    ----------
    population : int, optional
        The population size, by default 30
    Ts : float, optional
        Starting temperature, by default 10.0
    Tf : float, optional
        Final temperature, by default 0.1
    n_T_adj : int, optional
        Number of temperature adjustments, by default 10
    n_range_adj : int, optional
        Number of range adjustments, by default 10
    bin_size : int, optional
        Size of bins for temperature adjustment, by default 10
    start_range : float, optional
        Starting range for search, by default 1.0
    seed : Optional[int], optional
        Seed for random number generation, by default None

    Returns
    -------
    PygmoAlgorithm
        A configured instance of the Simulated Annealing optimizer.
    """
    return OptimizerTuple(
        base_class=PygmoAlgorithm,
        algorithm=pg.simulated_annealing,
        hyperparameters={
            'population': population,
            'Ts': Ts,
            'Tf': Tf,
            'n_T_adj': n_T_adj,
            'n_range_adj': n_range_adj,
            'bin_size': bin_size,
            'start_range': start_range,
            'seed': seed,
            **kwargs
        })


# =============================================================================


def xnes(population: int = 30,
         eta_mu: float = -1.0,
         eta_sigma: float = -1.0,
         eta_b: float = -1.0,
         sigma0: float = -1.0,
         ftol: float = 1e-06,
         xtol: float = 1e-06,
         force_bounds: bool = True,
         seed: Optional[int] = None,
         **kwargs) -> OptimizerTuple:
    """
    Exponential Natural Evolution Strategies (xNES) optimizer using pygmo.

    Parameters
    ----------
    population : int, optional
        The population size, by default 30
    eta_mu : float, optional
        Learning rate for the mean, by default -1.0
    eta_sigma : float, optional
        Learning rate for sigma, by default -1.0
    eta_b : float, optional
        Learning
    sigma0 : float, optional
        Initial sigma, by default -1.0
    ftol : float, optional
        Function tolerance, by default 1e-06
    xtol : float, optional
        Variable tolerance, by default 1e-06
    force_bounds : bool, optional
        If bounds should be enforced, by default True
    seed : Optional[int], optional
        Seed for random number generation, by default None

    Returns
    -------
    PygmoAlgorithm
        A configured instance of the xNES optimizer.
    """
    return OptimizerTuple(
        base_class=PygmoAlgorithm,
        algorithm=pg.xnes,
        hyperparameters={
            'population': population,
            'eta_mu': eta_mu,
            'eta_sigma': eta_sigma,
            'eta_b': eta_b,
            'sigma0': sigma0,
            'ftol': ftol,
            'xtol': xtol,
            'force_bounds': force_bounds,
            'seed': seed,
            'memory': True,
            'gen': 1,
            **kwargs
        })
