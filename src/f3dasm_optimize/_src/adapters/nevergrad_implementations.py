#                                                                       Modules
# =============================================================================

# Standard
from typing import Optional, Tuple

# Third-party
import autograd.numpy as np
import nevergrad as ng
from f3dasm import ExperimentData, ExperimentSample
from f3dasm.datageneration import DataGenerator
from f3dasm.optimization import Optimizer

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class NeverGradOptimizer(Optimizer):
    require_gradients: bool = False

    def __init__(self, algorithm_cls, population: int,
                 seed: Optional[int] = None,
                 **hyperparameters):
        self.algorithm_cls = algorithm_cls
        self.population = population
        self.seed = seed
        self.hyperparameters = hyperparameters

    def init(self, data: ExperimentData, data_generator: DataGenerator):
        self.data = data
        self.data_generator = data_generator
        p = ng.p.Array(shape=(len(self.data.domain),),
                       lower=self.data.domain.get_bounds()[:, 0],
                       upper=self.data.domain.get_bounds()[:, 1],
                       )

        p._set_random_state(np.random.RandomState(self.seed))

        self.algorithm = self.algorithm_cls(
            popsize=self.population,
            **self.hyperparameters)(p, budget=1e8)

    def update_step(self) -> Tuple[np.ndarray, None]:
        x = [self.algorithm.ask() for _ in range(
            self.population)]

        # Evaluate the candidates
        y = []
        for x_i in x:
            _x = ExperimentSample.from_numpy(input_array=x_i.value,
                                             domain=self.data.domain)
            experiment_sample = self.data_generator._run(
                _x,
                domain=self.data.domain)
            y.append(experiment_sample.to_numpy()[1])

        for x_tell, y_tell in zip(x, y):
            self.algorithm.tell(x_tell, y_tell)

        # return the data
        return np.vstack([x_.value for x_ in x]), np.array(y).ravel()
