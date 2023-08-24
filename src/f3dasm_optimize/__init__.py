"""
Some API information about the opitmizers
"""
#                                                                       Modules
# =============================================================================

# Standard
import sys
from itertools import chain
from os import path
from pathlib import Path
from typing import TYPE_CHECKING

# Local
from ._imports import _IntegrationModule

if TYPE_CHECKING:
    from ._all_optimizers import OPTIMIZERS
    from ._version import __version__
    from .adam import Adam, Adam_Parameters
    from .adamax import Adamax, Adamax_Parameters
    from .bayesianoptimization import (BayesianOptimization,
                                       BayesianOptimization_Parameters)
    from .cmaes import CMAES, CMAES_Parameters
    from .differential_evoluation_nevergrad import \
        DifferentialEvolution_Nevergrad
    from .differentialevolution import (DifferentialEvolution,
                                        DifferentialEvolution_Parameters)
    from .evosax_implementations import (EvoSaxCMAES, EvoSaxCMAES_Parameters,
                                         EvoSaxDE, EvoSaxPSO, EvoSaxSimAnneal)
    from .ftrl import Ftrl, Ftrl_Parameters
    from .mma import MMA, MMA_Parameters
    from .nadam import Nadam, Nadam_Parameters
    from .pso import PSO, PSO_Parameters
    from .pso_nevergrad import PSOConf
    from .rmsprop import RMSprop, RMSprop_Parameters
    from .sade import SADE, SADE_Parameters
    from .sea import SEA, SEA_Parameters
    from .sga import SGA, SGA_Parameters
    from .sgd import SGD, SGD_Parameters
    from .simulatedannealing import (SimulatedAnnealing,
                                     SimulatedAnnealing_Parameters)
    from .xnes import XNES, XNES_Parameters

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

_import_structure: dict = {
    "adam": ["Adam", "Adam_Parameters"],
    "adamax": ["Adamax", "Adamax_Parameters"],
    "bayesianoptimization": ["BayesianOptimization", "BayesianOptimization_Parameters"],
    "cmaes": ["CMAES", "CMAES_Parameters"],
    "differentialevolution": ["DifferentialEvolution", "DifferentialEvolution_Parameters"],
    "ftrl": ["Ftrl", "Ftrl_Parameters"],
    "nadam": ["Nadam", "Nadam_Parameters"],
    "pso": ["PSO", "PSO_Parameters"],
    "rmsprop": ["RMSprop", "RMSprop_Parameters"],
    "sade": ["SADE", "SADE_Parameters"],
    "sea": ["SEA", "SEA_Parameters"],
    "sga": ["SGA", "SGA_Parameters"],
    "sgd": ["SGD", "SGD_Parameters"],
    "simulatedannealing": ["SimulatedAnnealing", "SimulatedAnnealing_Parameters"],
    "xnes": ["XNES", "XNES_Parameters"],
    "mma": ["MMA", "MMA_Parameters"],
    "pso_nvergrad": ["PSOConf", "PSOConf_Parameters"],
    "differentialevolution_nevergrad": ["DifferentialEvolution_Nevergrad", "DifferentialEvolution_Nevergrad_Parameters"],
    "evosax_implementations": ["EvoSaxCMAES", "EvoSaxPSO", "EvoSaxSimAnneal", "EvoSaxDE", "EvoSaxCMAES_Parameters"],
    "_all_optimizers": ["OPTIMIZERS"],
    "_version": ["__version__"],
}

if not TYPE_CHECKING:
    class _LocalIntegrationModule(_IntegrationModule):
        __file__ = globals()["__file__"]
        __path__ = [path.dirname(__file__)]
        __all__ = list(chain.from_iterable(_import_structure.values()))
        _import_structure = _import_structure

    sys.modules[__name__] = _LocalIntegrationModule(__name__)
