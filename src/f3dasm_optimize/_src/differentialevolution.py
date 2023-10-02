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
class DifferentialEvolution_Parameters(OptimizerParameters):
    """Hyperparameters for DifferentialEvolution optimizer

    Args:
        population (int): _description_ (Default = 30)
        F (float): _description_ (Default = 0.8)
        CR (float): _description_ (Default = 0.9)
        variant (int): _description_ (Default = 2)
        ftol (float): _description_ (Default = 0.0)
        xtol (float): _description_ (Default = 0.0)
    """

    population: int = 30
    F: float = 0.8
    CR: float = 0.9
    variant: int = 2
    ftol: float = 0.0
    xtol: float = 0.0


class DifferentialEvolution(PygmoAlgorithm):
    "DifferentialEvolution optimizer implemented from pygmo"

    hyperparameters: DifferentialEvolution_Parameters = DifferentialEvolution_Parameters()

    def set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.de(
                gen=1,
                F=self.hyperparameters.F,
                CR=self.hyperparameters.CR,
                variant=self.hyperparameters.variant,
                ftol=self.hyperparameters.ftol,
                xtol=self.hyperparameters.xtol,
                seed=self.seed,
            )
        )

    def get_info(self) -> List[str]:
        return ['Fast', 'Global', 'Derivative-Free', 'Population-Based', 'Single-Solution']
