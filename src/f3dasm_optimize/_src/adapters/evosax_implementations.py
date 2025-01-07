#                                                                       Modules
# =============================================================================

# Standard
from typing import Optional, Tuple, Type

# Third-party
import jax
import numpy as np
from evosax import Strategy
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


class EvoSaxOptimizer(Optimizer):
    type: str = 'evosax'
    require_gradients: bool = False

    def __init__(self, algorithm_cls: Type[Strategy], population: int,
                 seed: Optional[int], **hyperparameters):

        if seed is None:
            seed = np.random.default_rng().integers(1e6)

        self.algorithm_cls = algorithm_cls
        self.population = population
        self.seed = seed
        self.hyperparameters = hyperparameters

    def init(self, data: ExperimentData, data_generator: DataGenerator):
        self.data = data
        self.data_generator = data_generator
        self.algorithm: Strategy = self.algorithm_cls(num_dims=len(
            data.domain), popsize=self.population, **self.hyperparameters)

        # Set algorithm
        _, rng_ask = jax.random.split(
            jax.random.PRNGKey(self.seed))
        self.evosax_param = self.algorithm.default_params
        self.evosax_param = self.evosax_param.replace(
            clip_min=data.domain.get_bounds()[
                0, 0], clip_max=self.data.domain.get_bounds()[0, 1])

        self.state = self.algorithm.initialize(
            rng_ask, self.evosax_param)

        # Set data
        x_init, y_init = self.data.get_n_best_output(
            self.population).to_numpy()

        self.state = self.algorithm.tell(
            x_init, y_init.ravel(), self.state, self.evosax_param)

    def update_step(self) -> Tuple[np.ndarray, np.ndarray]:
        _, rng_ask = jax.random.split(
            jax.random.PRNGKey(self.seed))

        # Ask for a set candidates
        x, state = self.algorithm.ask(
            rng_ask, self.state, self.evosax_param)

        # Evaluate the candidates
        y = []
        for x_i in np.array(x):
            _x = ExperimentSample.from_numpy(input_array=x_i,
                                             domain=self.data.domain)
            experiment_sample = self.data_generator._run(
                _x, domain=self.data.domain)
            y.append(experiment_sample.to_numpy()[1])

        y = np.array(y).ravel()

        # Update the strategy based on fitness
        self.state = self.algorithm.tell(
            x, y.ravel(), state, self.evosax_param)
        return np.array(x), y
