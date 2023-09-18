#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass
from typing import List

# Locals
from f3dasm import try_import
from .adapters.pygmo_implementations import PygmoAlgorithm
from f3dasm.optimization import OptimizerParameters

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
class SGA_Parameters(OptimizerParameters):
    """Hyperparameters for SGA optimizer"""

    cr: float = 0.9
    eta_c: float = 1.0
    m: float = 0.02
    param_m: float = 1.0
    param_s: float = 2
    crossover: str = "exponential"
    mutation: str = "polynomial"
    selection: str = "tournament"
    population: int = 30


class SGA(PygmoAlgorithm):
    """Simple Genetic Algorithm optimizer implemented from pygmo"""

    hyperparameters: SGA_Parameters = SGA_Parameters()

    def set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.sga(
                gen=1,
                cr=self.hyperparameters.cr,
                eta_c=self.hyperparameters.eta_c,
                m=self.hyperparameters.m,
                param_m=self.hyperparameters.param_m,
                param_s=self.hyperparameters.param_s,
                crossover=self.hyperparameters.crossover,
                mutation=self.hyperparameters.mutation,
                selection=self.hyperparameters.selection,
                seed=self.seed,
            )
        )

    def get_info(self) -> List[str]:
        return ['Fast', 'Population-Based']
