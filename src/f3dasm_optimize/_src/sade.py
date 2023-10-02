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
class SADE_Parameters(OptimizerParameters):
    """Hyperparameters for Self-adaptive Differential Evolution optimizer"""

    population: int = 30
    variant: int = 2
    variant_adptv: int = 1
    ftol: float = 0.0
    xtol: float = 0.0


class SADE(PygmoAlgorithm):
    "Self-adaptive Differential Evolution optimizer implemented from pygmo"

    hyperparameters: SADE_Parameters = SADE_Parameters()

    def set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.sade(
                gen=1,
                variant=self.hyperparameters.variant,
                variant_adptv=self.hyperparameters.variant_adptv,
                ftol=self.hyperparameters.ftol,
                xtol=self.hyperparameters.xtol,
                memory=True,
                seed=self.seed,
            )
        )

    def get_info(self) -> List[str]:
        return ['Fast', 'Population-Based', 'Single-Solution']
