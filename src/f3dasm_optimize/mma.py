#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass
from typing import List

# Locals
from f3dasm._imports import try_import
from f3dasm.optimization.optimizer import OptimizerParameters

from .adapters.pygmo_implementations import PygmoAlgorithm

# Third-party extension
with try_import('optimization') as _imports:
    import pygmo as pg


#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


@dataclass
class MMA_Parameters(OptimizerParameters):
    """Hyperparameters for MMA optimizer"""
    ...


class MMA(PygmoAlgorithm):
    """Method of Movinging Asymptotes optimizer implemented from pygmo, ported from NLOpt"""

    parameter: MMA_Parameters = MMA_Parameters()

    def set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.nlopt(
                solver='mma'
            )
        )

    def get_info(self) -> List[str]:
        return ['Stable', 'Local', 'Single-Solution']
