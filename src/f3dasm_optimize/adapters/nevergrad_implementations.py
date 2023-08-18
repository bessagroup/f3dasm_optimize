#                                                                       Modules
# =============================================================================

# Standard
from typing import Tuple

# Third-party core
import autograd.numpy as np
# Locals
# from f3dasm._imports import try_import
from f3dasm.optimization.optimizer import Optimizer

from .._protocol import Function

# # Third-party extension
# with try_import('optimization') as _imports:
#     import nevergrad as ng

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class NeverGradOptimizer(Optimizer):

    @staticmethod
    def _check_imports():
        ...
        # _imports.check()

    def update_step(self, function: Function) -> Tuple[np.ndarray, np.ndarray]:
        x = self.algorithm.ask()
        y: np.ndarray = function(x.value).ravel()[0]
        self.algorithm.tell(x, y)
        return x.value.reshape(1, -1), np.array(y).reshape(-1, 1)
