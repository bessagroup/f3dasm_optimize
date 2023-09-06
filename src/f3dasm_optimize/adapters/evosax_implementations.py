from typing import Tuple

import numpy as np
from f3dasm._imports import try_import
from f3dasm.datageneration.functions import Function
from f3dasm.optimization import Optimizer


# Third-party extension
with try_import('optimization') as _imports:
    import jax
    from evosax import Strategy
    from jax._src.typing import Array


class EvoSaxOptimizer(Optimizer):
    # evosax_algorithm: Strategy = None

    def _construct_model(self, function: Function):
        self.algorithm: Strategy = self.evosax_algorithm(
            num_dims=function.dimensionality, popsize=self.parameter.population)
        self.evosax_param = self.algorithm.default_params
        self.evosax_param = self.evosax_param.replace(clip_min=self.data.domain.get_bounds()[
            0, 0], clip_max=self.data.domain.get_bounds()[0, 1])

        self.state = self.algorithm.initialize(self.seed, self.evosax_param)

        x_init = self.data.get_n_best_input_parameters_numpy(self.parameter.population)
        y_init = function(x_init).ravel()

        self.state = self.algorithm.tell(x_init, y_init, self.state, self.evosax_param)

    def set_seed(self, seed: int) -> Array:
        self.seed = jax.random.PRNGKey(seed)

    def reset(self):
        self._check_imports()

        if self.name is None:
            self.name = self.__class__.__name__

        self.init_parameters()
        self._set_hyperparameters()
        self.set_algorithm()

    def update_step(self, function: Function) -> Tuple[np.ndarray, np.ndarray]:
        _, rng_ask = jax.random.split(self.seed)
        # Ask for a set candidates
        x, state = self.algorithm.ask(rng_ask, self.state, self.evosax_param)
        # Evaluate the candidates
        y = function(x)
        # Update the strategy based on fitness
        self.state = self.algorithm.tell(x, y.ravel(), state, self.evosax_param)
        return x, y

    def iterate(self, iterations: int, function: Function):
        """Calls the update_step function multiple times.

        Parameters
        ----------
        iterations
            number of iterations
        function
            objective function to evaluate
        """
        self._construct_model(function)

        self._check_number_of_datapoints()

        for _ in range(_number_of_updates(iterations, population=self.parameter.population)):
            x, y = self.update_step(function=function)
            self.add_iteration_to_data(x, y)

        # Remove overiterations
        self.data.remove_rows_bottom(_number_of_overiterations(
            iterations, population=self.parameter.population))

def _number_of_updates(iterations: int, population: int):
    """Calculate number of update steps to acquire the correct number of iterations

    Parameters
    ----------
    iterations
        number of desired iteration steps
    population
        the population size of the optimizer

    Returns
    -------
        number of consecutive update steps
    """
    return iterations // population + (iterations % population > 0)


def _number_of_overiterations(iterations: int, population: int) -> int:
    """Calculate the number of iterations that are over the iteration limit

    Parameters
    ----------
    iterations
        number of desired iteration steos
    population
        the population size of the optimizer

    Returns
    -------
        number of iterations that are over the limit
    """
    overiterations: int = iterations % population
    if overiterations == 0:
        return overiterations
    else:
        return population - overiterations
