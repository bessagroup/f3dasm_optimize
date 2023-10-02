#                                                                       Modules
# =============================================================================

from typing import Tuple

# Third-party core
# Locals
# from f3dasm._imports import try_import
import autograd.numpy as np
from f3dasm.optimization import Optimizer

from .._protocol import DataGenerator, ExperimentSample

# # Third-party extension
# with try_import('optimization') as _imports:
#     import nevergrad as ng

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class NeverGradOptimizer(Optimizer):

    @staticmethod
    def _check_imports():
        ...

    def update_step(self, data_generator: DataGenerator) -> Tuple[np.ndarray, None]:
        x = [self.algorithm.ask() for _ in range(self.hyperparameters.population)]

        # Evaluate the candidates
        y = []
        for x_i in x:
            experiment_sample = ExperimentSample.from_numpy(input_array=x_i)
            data_generator.run(experiment_sample)
            y.append(experiment_sample.to_numpy()[1])

        for x_tell, y_tell in zip(x, y):
            self.algorithm.tell(x_tell, y_tell)

        # return the data
        return np.vstack([x_.value for x_ in x]), np.atleast_2d(np.array(y))
