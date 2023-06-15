"""
f3dasm_optimize : Optimization extenstion for f3dasm
"""
#                                                                       Modules
# =============================================================================

# Standard
from typing import List

# Third-party
from f3dasm.optimization.optimizer import Optimizer

# Local
from ._all_optimizers import OPTIMIZERS
from .adam import Adam
from .adamax import Adamax
from .bayesianoptimization import BayesianOptimization
from .cg import CG
from .cmaes import CMAES
from .differentialevolution import DifferentialEvolution
from .ftrl import Ftrl
from .lbfgsb import LBFGSB
from .nadam import Nadam
from .neldermead import NelderMead
from .pso import PSO
from .randomsearch import RandomSearch
from .rmsprop import RMSprop
from .run_optimization import run_multiple_realizations
from .sade import SADE
from .sea import SEA
from .sga import SGA
from .sgd import SGD
from .simulatedannealing import SimulatedAnnealing
from .utils import (create_optimizer_from_dict, create_optimizer_from_json,
                    find_optimizer)
from .xnes import XNES

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
