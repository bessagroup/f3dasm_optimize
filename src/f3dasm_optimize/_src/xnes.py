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
class XNES_Parameters(OptimizerParameters):
    """Hyperparameters for XNES optimizer


    """

    population: int = 30
    eta_mu: float = -1.0
    eta_sigma: float = -1.0
    eta_b: float = -1.0
    sigma0: float = -1.0
    ftol: float = 1e-06
    xtol: float = 1e-06


class XNES(PygmoAlgorithm):
    """XNES optimizer implemented from pygmo"""

    hyperparameters: XNES_Parameters = XNES_Parameters()

    def set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.xnes(
                gen=1,
                eta_mu=self.hyperparameters.eta_mu,
                eta_sigma=self.hyperparameters.eta_sigma,
                eta_b=self.hyperparameters.eta_b,
                sigma0=self.hyperparameters.sigma0,
                ftol=self.hyperparameters.ftol,
                xtol=self.hyperparameters.xtol,
                memory=True,
                force_bounds=self.hyperparameters.force_bounds,
                seed=self.seed,
            )
        )

    def get_info(self) -> List[str]:
        return ['Stable', 'Global', 'Population-Based']
