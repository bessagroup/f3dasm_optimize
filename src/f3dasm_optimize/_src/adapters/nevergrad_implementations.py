#                                                                       Modules
# =============================================================================

# Standard
from typing import Optional, Tuple

# Third-party
import autograd.numpy as np
import nevergrad as ng

# Local
from .._protocol import DataGenerator, Domain, Optimizer

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

    def __init__(self, domain: Domain, data_generator: DataGenerator,
                 algorithm, population: int, seed: Optional[int] = None,
                 **hyperparameters):
        self.domain = domain
        self.data_generator = data_generator
        self.population = population
        self.seed = seed
        p = ng.p.Array(shape=(len(self.domain),),
                       lower=self.domain.get_bounds()[:, 0],
                       upper=self.domain.get_bounds()[:, 1],
                       )

        p._set_random_state(np.random.RandomState(seed))

        self.algorithm = algorithm(popsize=population,
                                   **hyperparameters)(p, budget=1e8)

    def update_step(self) -> Tuple[np.ndarray, None]:
        x = [self.algorithm.ask() for _ in range(
            self.population)]

        # Evaluate the candidates
        y = []
        for x_i in x:
            experiment_sample = self.data_generator._run(x_i.value,
                                                         domain=self.domain)
            y.append(experiment_sample.to_numpy()[1])

        for x_tell, y_tell in zip(x, y):
            self.algorithm.tell(x_tell, y_tell)

        # return the data
        return np.vstack([x_.value for x_ in x]), np.array(y).ravel()
