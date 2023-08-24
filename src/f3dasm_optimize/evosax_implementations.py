from dataclasses import dataclass

from f3dasm._imports import try_import
from f3dasm.optimization.optimizer import OptimizerParameters

from .adapters.evosax_implementations import EvoSaxOptimizer

# Third-party extension
with try_import('optimization') as _imports:
    from evosax import CMA_ES, PSO, SimAnneal, DE


@dataclass
class EvoSaxCMAES_Parameters(OptimizerParameters):
    """Hyperparameters for EvoSaxCMAES optimizer"""

    population: int = 30


class EvoSaxCMAES(EvoSaxOptimizer):
    parameter: EvoSaxCMAES_Parameters = EvoSaxCMAES_Parameters()
    evosax_algorithm = CMA_ES


class EvoSaxPSO(EvoSaxOptimizer):
    parameter: EvoSaxCMAES_Parameters = EvoSaxCMAES_Parameters()
    evosax_algorithm = PSO


class EvoSaxSimAnneal(EvoSaxOptimizer):
    parameter: EvoSaxCMAES_Parameters = EvoSaxCMAES_Parameters()
    evosax_algorithm = SimAnneal

class EvoSaxDE(EvoSaxOptimizer):
    parameter: EvoSaxCMAES_Parameters = EvoSaxCMAES_Parameters()
    evosax_algorithm = DE
