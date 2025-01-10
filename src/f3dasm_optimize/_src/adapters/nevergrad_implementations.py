#                                                                       Modules
# =============================================================================

# Standard
from typing import Optional

# Third-party
import autograd.numpy as np
import nevergrad as ng
from f3dasm import Block, ExperimentData

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class NeverGradOptimizer(Block):
    require_gradients: bool = False

    def __init__(self, algorithm_cls, population: int,
                 seed: Optional[int] = None,
                 **hyperparameters):
        self.algorithm_cls = algorithm_cls
        self.population = population
        self.seed = seed
        self.hyperparameters = hyperparameters

    def arm(self, data: ExperimentData):
        self.data = data
        p = ng.p.Array(shape=(len(self.data.domain),),
                       lower=self.data.domain.get_bounds()[:, 0],
                       upper=self.data.domain.get_bounds()[:, 1],
                       )

        p._set_random_state(np.random.RandomState(self.seed))

        self.algorithm = self.algorithm_cls(
            popsize=self.population,
            **self.hyperparameters)(p, budget=1e8)

    def call(self, **kwargs) -> ExperimentData:

        # Get the last candidates
        xx, yy = self.data[-self.population:].to_numpy()

        for x_tell, y_tell in zip(xx, yy):
            self.algorithm.tell(
                self.algorithm.parametrization.spawn_child(x_tell), y_tell)

        x = [self.algorithm.ask() for _ in range(
            self.population)]

        return type(self.data)(input_data=np.vstack([x_.value for x_ in x]),
                               domain=self.data.domain,
                               project_dir=self.data.project_dir)
