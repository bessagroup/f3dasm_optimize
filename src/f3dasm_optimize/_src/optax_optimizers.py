#                                                                       Modules
# =============================================================================

# Third-party
from typing import Optional

import optax

# Local
from ._protocol import OptimizerTuple
from .adapters.optax_implementations import OptaxOptimizer

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


def adam(learning_rate: float = 0.001, beta_1: float = 0.9,
         beta_2: float = 0.999, epsilon: float = 1e-07, eps_root: float = 0.0,
         seed: Optional[int] = None, **kwargs) -> OptimizerTuple:
    """
    Adam optimizer.
    Adapted from the Optax library.

    Parameters
    ----------
    learning_rate : float, optional
        The learning rate, by default 0.001.
    beta_1 : float, optional
        Exponential decay rate for the first moment estimates, by default 0.9.
    beta_2 : float, optional
        Exponential decay rate for the second moment estimates,
        by default 0.999.
    epsilon : float, optional
        A small constant for numerical stability, by default 1e-07.
    eps_root : float, optional
        A small constant for numerical stability, by default 0.0.
    seed : int, optional
        Random seed, by default None.

    Returns
    -------
    OptimizerTuple
        OptimizerTuple object.
    """

    return OptimizerTuple(
        base_class=OptaxOptimizer,
        algorithm=optax.adam,
        hyperparameters={
            'learning_rate': learning_rate,
            'beta_1': beta_1,
            'beta_2': beta_2,
            'epsilon': epsilon,
            'eps_root': eps_root,
            'seed': seed,
            **kwargs
        }
    )

# =============================================================================


def sgd(learning_rate: float = 0.01, momentum: float = 0.0,
        nesterov: bool = False, seed: Optional[int] = None, **kwargs
        ) -> OptimizerTuple:
    """
    Stochastic Gradient Descent (SGD) optimizer.
    Adapted from the Optax library.

    Parameters
    ----------
    learning_rate : float, optional
        The learning rate, by default 0.01.
    momentum : float, optional
        Momentum parameter, by default 0.0.
    nesterov : bool, optional
        Use Nesterov momentum, by default False.
    seed : int, optional
        Random seed, by default None.

    Returns
    -------
    OptimizerTuple
        OptimizerTuple object.
    """

    return OptimizerTuple(
        base_class=OptaxOptimizer,
        algorithm=optax.sgd,
        hyperparameters={
            'learning_rate': learning_rate,
            'momentum': momentum,
            'nesterov': nesterov,
            'seed': seed,
            **kwargs
        }
    )

# =============================================================================
