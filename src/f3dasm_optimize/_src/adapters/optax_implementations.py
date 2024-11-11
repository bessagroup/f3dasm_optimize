#                                                                       Modules
# =============================================================================

# Standard
from typing import Optional, Tuple

# Third-party
import jax.numpy as jnp
import numpy as onp
import optax

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


class OptaxOptimizer(Optimizer):
    require_gradients: bool = True

    def __init__(self, domain: Domain, data_generator: DataGenerator,
                 algorithm, seed: Optional[int], **hyperparameters):
        self.domain = domain
        self.data_generator = data_generator
        self.seed = seed
        self.algorithm = algorithm(**hyperparameters)

    def init(self):
        self.grad_f = lambda params: jnp.array(
            self.data_generator.dfdx(onp.array(params)))
        self.params = jnp.array(self.data.get_experiment_sample(
            self.data.index[-1]).to_numpy()[0])
        self.opt_state = self.algorithm.init(self.params)

    def update_step(self) -> Tuple[onp.ndarray, onp.ndarray]:
        updates, self.opt_state = self.algorithm.update(
            self.grad_f(self.params), self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        self.params = jnp.clip(self.params, self.domain.get_bounds()[
                               :, 0], self.domain.get_bounds()[:, 1])
        return onp.atleast_2d(self.params), None
