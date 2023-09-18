#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass
from typing import List

# Locals
from f3dasm import try_import
from f3dasm.optimization import OptimizerParameters

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
class CMAES_Parameters(OptimizerParameters):
    """Hyperparameters for CMAES optimizer"""

    population: int = 30


class CMAES(PygmoAlgorithm):
    """Covariance Matrix Adaptation Evolution Strategy optimizer implemented from pygmo"""

    hyperparameters: CMAES_Parameters = CMAES_Parameters()

    def set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.cmaes(
                gen=1,
                memory=True,
                seed=self.seed,
                force_bounds=self.hyperparameters.force_bounds,
            )
        )

    def get_info(self) -> List[str]:
        return ['Stable', 'Global', 'Population-Based']
