#                                                                       Modules
# =============================================================================

# Standard
from typing import List

# Locals
from f3dasm.optimization import Optimizer

from . import (adam, adamax, cmaes, differential_evoluation_nevergrad,
               differentialevolution, ftrl, mma, nadam, pso, pso_nevergrad,
               rmsprop, sade, sea, sga, sgd, simulatedannealing, xnes)

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

OPTIMIZERS: List[Optimizer] = []

if adam._imports.is_successful():
    OPTIMIZERS.append(adam.Adam)

if adamax._imports.is_successful():
    OPTIMIZERS.append(adamax.Adamax)

# COMMENT THIS BECAUSE BAYESIAN OPTIMIZATION IS TOO SLOW TO TEST PROPERLY NOW
# if bayesianoptimization._imports.is_successful():
#     OPTIMIZERS.append(bayesianoptimization.BayesianOptimization)

if cmaes._imports.is_successful():
    OPTIMIZERS.append(cmaes.CMAES)

if differentialevolution._imports.is_successful():
    OPTIMIZERS.append(differentialevolution.DifferentialEvolution)

if ftrl._imports.is_successful():
    OPTIMIZERS.append(ftrl.Ftrl)

if nadam._imports.is_successful():
    OPTIMIZERS.append(nadam.Nadam)

if pso._imports.is_successful():
    OPTIMIZERS.append(pso.PSO)

if rmsprop._imports.is_successful():
    OPTIMIZERS.append(rmsprop.RMSprop)

if sade._imports.is_successful():
    OPTIMIZERS.append(sade.SADE)

if sea._imports.is_successful():
    OPTIMIZERS.append(sea.SEA)

if sga._imports.is_successful():
    OPTIMIZERS.append(sga.SGA)

if sgd._imports.is_successful():
    OPTIMIZERS.append(sgd.SGD)

if simulatedannealing._imports.is_successful():
    OPTIMIZERS.append(simulatedannealing.SimulatedAnnealing)

if xnes._imports.is_successful():
    OPTIMIZERS.append(xnes.XNES)

# if mma._imports.is_successful():
#     OPTIMIZERS.append(mma.MMA)

# if pso_nevergrad._imports.is_successful():
#     OPTIMIZERS.append(pso_nevergrad.PSOConf)

# if differential_evoluation_nevergrad._imports.is_successful():
#     OPTIMIZERS.append(differential_evoluation_nevergrad.DifferentialEvolution_Nevergrad)
