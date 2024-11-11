#                                                                       Modules
# =============================================================================

# Standard
from typing import Optional

# Third party
import optuna

from f3dasm_optimize._src._protocol import Domain, OptimizerTuple

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


class TPESampler(OptunaOptimizer):
    require_gradients: bool = False

    def __init__(self, domain: Domain, seed: Optional[int] = None, **kwargs):
        super().__init__(domain)
        self.seed = seed
        self._set_algorithm()

    def _set_algorithm(self):
        self.algorithm = optuna.create_study(
            sampler=optuna.samplers.TPESampler(
                seed=self.seed))


def tpe_sampler(seed: Optional[int] = None) -> OptimizerTuple:
    """
    Tree-structured Parzen Estimator (TPE) sampler.
    Adapted from the optuna library

    Parameters
    ----------
    seed : int, optional
        Random seed, by default None.

    Returns
    -------
    OptimizerTuple
        OptimizerTuple object.
    """

    return OptimizerTuple(
        optimizer=TPESampler,
        hyperparameters={
            'seed': seed
        }
    )
