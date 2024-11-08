from typing import ClassVar, NamedTuple, Optional

from f3dasm.design import Domain

from .evosax_optimizers import EvoSaxCMAES
from .optimizer import Optimizer


class OptimizerTuple(NamedTuple):
    optimizer: Optimizer
    hyperparameters: dict

    def tuple_to_optimizer(self, domain: Domain) -> Optimizer:
        return self.optimizer(domain=domain, **self.hyperparameters)


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
