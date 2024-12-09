#                                                                       Modules
# =============================================================================

# Standard
from typing import Optional

import optuna

# Third party
from f3dasm_optimize._src._protocol import Optimizer

# Local
from .adapters.optuna_implementations import OptunaOptimizer

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


def tpe_sampler(seed: Optional[int] = None, **kwargs) -> Optimizer:
    """
    Tree-structured Parzen Estimator (TPE) sampler.
    Adapted from the optuna library

    Parameters
    ----------
    seed : int, optional
        Random seed, by default None.

    Returns
    -------
    Optimizer
        Optimizer object.
    """

    return OptunaOptimizer(
        algorithm_cls=optuna.samplers.TPESampler,
        seed=seed,
        **kwargs
    )
