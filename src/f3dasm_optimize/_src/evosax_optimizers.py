#                                                                       Modules
# =============================================================================

# Standard
from typing import Optional

# Third-party
from evosax import BIPOP_CMA_ES, CMA_ES, DE, PSO, SimAnneal

# Local
from ._protocol import Domain, OptimizerTuple
from .adapters.evosax_implementations import EvoSaxOptimizer

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class EvoSaxCMAES(EvoSaxOptimizer):
    require_gradients: bool = False

    def __init__(
        self, domain: Domain, population: int = 30, seed: Optional[int] = None,
            **kwargs):
        super().__init__(
            domain=domain, population=population, seed=seed)
        self.evosax_algorithm = CMA_ES
        self._set_algorithm()


def cmaes(population: int = 30,
          seed: Optional[int] = None) -> OptimizerTuple:
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
        optimizer=EvoSaxCMAES,
        hyperparameters={'population': population, 'seed': seed})

# =============================================================================


class EvoSaxPSO(EvoSaxOptimizer):
    require_gradients: bool = False

    def __init__(
        self, domain: Domain, population: int = 30, seed: Optional[int] = None,
            **kwargs):
        super().__init__(
            domain=domain, population=population, seed=seed, **kwargs)
        self.evosax_algorithm = PSO
        self._set_algorithm()


def pso(population: int = 30,
        seed: Optional[int] = None) -> OptimizerTuple:
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
        optimizer=EvoSaxPSO,
        hyperparameters={'population': population, 'seed': seed})

# =============================================================================


class EvoSaxSimAnneal(EvoSaxOptimizer):
    require_gradients: bool = False

    def __init__(
        self, domain: Domain, population: int = 30, seed: Optional[int] = None,
            **kwargs):
        super().__init__(
            domain=domain, population=population, seed=seed)
        self.evosax_algorithm = SimAnneal
        self._set_algorithm()


def simanneal(population: int = 30,
              seed: Optional[int] = None) -> OptimizerTuple:
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
        optimizer=EvoSaxSimAnneal,
        hyperparameters={'population': population, 'seed': seed})

# =============================================================================


class EvoSaxDE(EvoSaxOptimizer):
    require_gradients: bool = False

    def __init__(
        self, domain: Domain, population: int = 30, seed: Optional[int] = None,
            **kwargs):
        super().__init__(
            domain=domain, population=population, seed=seed)
        self.evosax_algorithm = DE
        self._set_algorithm()


def de(population: int = 30,
       seed: Optional[int] = None) -> OptimizerTuple:
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
        optimizer=EvoSaxDE,
        hyperparameters={'population': population, 'seed': seed})

# =============================================================================


class EvoSaxBIPOPCMAES(EvoSaxOptimizer):
    require_gradients: bool = False

    def __init__(
        self, domain: Domain, population: int = 30, seed: Optional[int] = None,
            **kwargs):
        super().__init__(
            domain=domain, population=population, seed=seed)
        self.evosax_algorithm = BIPOP_CMA_ES
        self._set_algorithm()


def bipopcmaes(population: int = 30,
               seed: Optional[int] = None) -> OptimizerTuple:
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
        optimizer=EvoSaxBIPOPCMAES,
        hyperparameters={'population': population, 'seed': seed})

# =============================================================================
