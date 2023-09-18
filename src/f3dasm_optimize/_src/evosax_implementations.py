from dataclasses import dataclass

from f3dasm import try_import
from f3dasm.optimization import OptimizerParameters

from .adapters.evosax_implementations import EvoSaxOptimizer

# Third-party extension
with try_import('optimization') as _imports:
    from evosax import CMA_ES, DE, PSO, SimAnneal


@dataclass
class EvoSaxCMAES_Parameters(OptimizerParameters):
    """Hyperparameters for EvoSaxCMAES optimizer"""

    population: int = 30


class EvoSaxCMAES(EvoSaxOptimizer):
    hyperparameters: EvoSaxCMAES_Parameters = EvoSaxCMAES_Parameters()
    evosax_algorithm = CMA_ES


class EvoSaxPSO(EvoSaxOptimizer):
    hyperparameters: EvoSaxCMAES_Parameters = EvoSaxCMAES_Parameters()
    evosax_algorithm = PSO


class EvoSaxSimAnneal(EvoSaxOptimizer):
    hyperparameters: EvoSaxCMAES_Parameters = EvoSaxCMAES_Parameters()
    evosax_algorithm = SimAnneal

class EvoSaxDE(EvoSaxOptimizer):
    hyperparameters: EvoSaxCMAES_Parameters = EvoSaxCMAES_Parameters()
    evosax_algorithm = DE
