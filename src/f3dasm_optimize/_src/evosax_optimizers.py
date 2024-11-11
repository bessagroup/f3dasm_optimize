#                                                                       Modules
# =============================================================================

# Standard
from typing import Optional

# Third-party
from evosax import BIPOP_CMA_ES, CMA_ES, DE, PSO, SimAnneal

# Local
from ._protocol import OptimizerTuple
from .adapters.evosax_implementations import EvoSaxOptimizer

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


def cmaes(population: int = 30,
          seed: Optional[int] = None, **kwargs) -> OptimizerTuple:
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimizer.
    Adapated from the EvoSax library.

    Parameters
    ----------
    population : int, optional
        The number of individuals in the population, by default 30
    seed : Optional[int], optional
        The seed for the random number generator, by default None

    Returns
    -------
    OptimizerTuple
        A tuple containing the optimizer and its hyperparameters
    """
    return OptimizerTuple(
        base_class=EvoSaxOptimizer,
        algorithm=CMA_ES,
        hyperparameters={'population': population, 'seed': seed, **kwargs})

# =============================================================================


def pso(population: int = 30,
        seed: Optional[int] = None, **kwargs) -> OptimizerTuple:
    """
    Particle Swarm Optimization (PSO) optimizer.
    Adapted from the EvoSax library.

    Parameters
    ----------
    population : int, optional
        The number of individuals in the population, by default 30
    seed : Optional[int], optional
        The seed for the random number generator, by default None

    Returns
    -------
    OptimizerTuple
        A tuple containing the optimizer and its hyperparameters
    """
    return OptimizerTuple(
        base_class=EvoSaxOptimizer,
        algorithm=PSO,
        hyperparameters={'population': population, 'seed': seed, **kwargs})

# =============================================================================


def simanneal(population: int = 30,
              seed: Optional[int] = None, **kwargs) -> OptimizerTuple:
    """
    Simulated Annealing (SimAnneal) optimizer.
    Adapted from the EvoSax library.

    Parameters
    ----------
    population : int, optional
        The number of individuals in the population, by default 30
    seed : Optional[int], optional
        The seed for the random number generator, by default None

    Returns
    -------
    OptimizerTuple
        A tuple containing the optimizer and its hyperparameters
    """
    return OptimizerTuple(
        base_class=EvoSaxOptimizer,
        algorithm=SimAnneal,
        hyperparameters={'population': population, 'seed': seed, **kwargs})

# =============================================================================


def de(population: int = 30,
       seed: Optional[int] = None, **kwargs) -> OptimizerTuple:
    """
    Differential Evolution (DE) optimizer.
    Adapted from the EvoSax library.

    Parameters
    ----------
    population : int, optional
        The number of individuals in the population, by default 30
    seed : Optional[int], optional
        The seed for the random number generator, by default None

    Returns
    -------
    OptimizerTuple
        A tuple containing the optimizer and its hyperparameters
    """
    return OptimizerTuple(
        base_class=EvoSaxOptimizer,
        algorithm=DE,
        hyperparameters={'population': population, 'seed': seed, **kwargs})

# =============================================================================


def bipopcmaes(population: int = 30,
               seed: Optional[int] = None, **kwargs) -> OptimizerTuple:
    """
    BIPOP-CMA-ES optimizer.
    Adapted from the EvoSax library.

    Parameters
    ----------
    population : int, optional
        The number of individuals in the population, by default 30
    seed : Optional[int], optional
        The seed for the random number generator, by default None

    Returns
    -------
    OptimizerTuple
        A tuple containing the optimizer and its hyperparameters
    """
    return OptimizerTuple(
        base_class=EvoSaxOptimizer,
        algorithm=BIPOP_CMA_ES,
        hyperparameters={'population': population, 'seed': seed, **kwargs})

# =============================================================================
