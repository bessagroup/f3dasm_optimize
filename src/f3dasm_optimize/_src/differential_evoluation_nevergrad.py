#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass

# Locals
from f3dasm import try_import
from f3dasm.optimization import OptimizerParameters

from .adapters.nevergrad_implementations import NeverGradOptimizer

# Third-party extension
with try_import('optimization') as _imports:
    import nevergrad as ng


#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


@dataclass
class DifferentialEvolution_Nevergrad_Parameters(OptimizerParameters):
    population: int = 30
    initialization: str = 'parametrization'
    scale: float = 1.0
    recommendation: str = 'optimistic'
    crossover: float = 0.5
    F1: float = 0.8
    F2: float = 0.8


class DifferentialEvolution_Nevergrad(NeverGradOptimizer):

    hyperparameters: DifferentialEvolution_Nevergrad_Parameters = DifferentialEvolution_Nevergrad_Parameters()

    def set_algorithm(self):
        p = ng.p.Array(shape=(len(self.data.domain),),
                       lower=self.data.domain.get_bounds()[:, 0], upper=self.data.domain.get_bounds()[:, 1])
        self.algorithm = ng.optimizers.DifferentialEvolution(initialization=self.hyperparameters.initialization,
                                                             popsize=self.hyperparameters.population,
                                                             scale=self.hyperparameters.scale,
                                                             recommendation=self.hyperparameters.recommendation,
                                                             crossover=self.hyperparameters.crossover,
                                                             F1=self.hyperparameters.F1,
                                                             F2=self.hyperparameters.F2)(p, budget=1e8)
