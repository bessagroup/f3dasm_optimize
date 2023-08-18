#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass

# Locals
from f3dasm._imports import try_import
from f3dasm.optimization.optimizer import OptimizerParameters

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
class PSOConf_Parameters(OptimizerParameters):
    population: int = 40
    transform: str = 'identity'
    omega: float = 0.7213475204444817
    phip: float = 1.1931471805599454
    phig: float = 1.1931471805599454
    qo: bool = False
    sqo: bool = False
    so: bool = False


class PSOConf(NeverGradOptimizer):

    parameter: PSOConf_Parameters = PSOConf_Parameters()

    def set_algorithm(self):
        p = ng.p.Array(shape=(len(self.data.domain),),
                       lower=self.data.domain.get_bounds()[:, 0], upper=self.data.domain.get_bounds()[:, 1])
        self.algorithm = ng.optimizers.ConfPSO(transform=self.parameter.transform,
                                               popsize=self.parameter.population,
                                               omega=self.parameter.omega,
                                               phip=self.parameter.phip,
                                               phig=self.parameter.phig,
                                               qo=self.parameter.qo,
                                               sqo=self.parameter.sqo,
                                               so=self.parameter.so)(p, budget=1e8)
