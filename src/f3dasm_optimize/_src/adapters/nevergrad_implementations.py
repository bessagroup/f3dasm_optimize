#                                                                       Modules
# =============================================================================


# Third-party core
# Locals
# from f3dasm._imports import try_import
import autograd.numpy as np
from f3dasm.design import ExperimentData
from f3dasm.optimization import Optimizer

from .._protocol import DataGenerator

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

    def update_step(self, data_generator: DataGenerator) -> ExperimentData:
        x = [self.algorithm.ask() for _ in range(self.hyperparameters.population)]

        # Evaluate the candidates
        x_experimentdata = ExperimentData.from_numpy(domain=self.domain,
                                                     input_array=np.vstack([x_.value for x_ in x]))
        x_experimentdata.run(data_generator.run)

        _, y = x_experimentdata.to_numpy()
        for x_tell, y_tell in zip(x, y):
            self.algorithm.tell(x_tell, y_tell)

        # return the data
        return ExperimentData.from_numpy(domain=self.domain,
                                         input_array=np.vstack([x_.value for x_ in x]),
                                         output_array=y)
