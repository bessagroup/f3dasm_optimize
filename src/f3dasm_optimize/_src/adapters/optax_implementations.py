#                                                                       Modules
# =============================================================================

# Standard
from typing import Callable, Optional

# Third-party
import jax.numpy as jnp
import numpy as onp
import optax
from f3dasm import Block, ExperimentData

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class OptaxOptimizer(Block):
    require_gradients: bool = True

    def __init__(self, algorithm_cls, seed: Optional[int], **hyperparameters):
        self.algorithm_cls = algorithm_cls
        self.seed = seed
        self.hyperparameters = hyperparameters

    def arm(self, data: ExperimentData):
        self.data = data

        # Set algorithm
        self.algorithm = self.algorithm_cls(**self.hyperparameters)

        # Set data
        x = self.data[-1].to_numpy()[0].ravel()

        self.opt_state = self.algorithm.init(jnp.array(x))

    def call(self, grad_fn: Callable, **kwargs) -> ExperimentData:
        # Set data
        x = self.data[-1].to_numpy()[0].ravel()

        def grad_f(params):
            return jnp.array(
                grad_fn(onp.array(params)))

        updates, self.opt_state = self.algorithm.update(
            grad_f(x), self.opt_state)
        new_x = optax.apply_updates(jnp.array(x), updates)
        new_x = jnp.clip(new_x, self.data.domain.get_bounds()[
            :, 0], self.data.domain.get_bounds()[:, 1])

        return type(self.data)(input_data=onp.atleast_2d(new_x),
                               domain=self.data.domain,
                               project_dir=self.data.project_dir)
