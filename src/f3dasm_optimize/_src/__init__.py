#                                                                       Modules
# =============================================================================

# Standard

from ._imports import try_import
from ._version import __version__
from .evosax_optimizers import (EvoSaxBIPOPCMAES, EvoSaxCMAES, EvoSaxDE,
                                EvoSaxPSO, EvoSaxSimAnneal)
from .nevergrad_optimizers import PSO, NevergradDE

with try_import() as _imports:
    from .pygmo_optimizers import (CMAES, SADE, SEA, SGA, XNES,
                                   DifferentialEvolution, PygmoPSO,
                                   SimulatedAnnealing)
# from .bayesianoptimization import BayesianOptimization
from .optuna_optimizers import TPESampler
from .tensorflow_optimizers import SGD, Adam, Adamax, Ftrl, Nadam, RMSprop

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


_OPTIMIZERS = [Adam, Adamax, Ftrl, Nadam, RMSprop, SGD, EvoSaxPSO,
               EvoSaxSimAnneal, EvoSaxDE, EvoSaxCMAES, NevergradDE, PSO, TPESampler, EvoSaxBIPOPCMAES]

if _imports.is_successful():
    _OPTIMIZERS.extend([CMAES, PygmoPSO, SADE, SEA, SGA, XNES,
                        DifferentialEvolution, SimulatedAnnealing])


__all__ = [
    'Adam',
    'Adamax',
    'CMAES',
    'DifferentialEvolution',
    'EvoSaxCMAES',
    'EvoSaxDE',
    'EvoSaxPSO',
    'EvoSaxSimAnneal',
    'EvoSaxBIPOPCMAES',
    'Ftrl',
    'MMA',
    'Nadam',
    'NevergradDE',
    'PSO',
    'PygmoPSO',
    'RMSprop',
    'SADE',
    'SEA',
    'SGA',
    'SGD',
    'SimulatedAnnealing',
    'XNES',
    'TPESampler',
    '__version__',
]
