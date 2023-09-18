
from f3dasm import try_import
from f3dasm import ExperimentData
from f3dasm.optimization import Optimizer

from .._protocol import DataGenerator

# Third-party extension
with try_import('optimization') as _imports:
    import jax
    from evosax import Strategy


class EvoSaxOptimizer(Optimizer):
    type: str = 'evosax'
    # evosax_algorithm: Strategy = None

    def _construct_model(self, data_generator: DataGenerator):
        _, rng_ask = jax.random.split(jax.random.PRNGKey(self.seed))
        self.algorithm: Strategy = self.evosax_algorithm(
            num_dims=len(self.domain), popsize=self.hyperparameters.population)
        self.evosax_param = self.algorithm.default_params
        self.evosax_param = self.evosax_param.replace(clip_min=self.data.domain.get_bounds()[
            0, 0], clip_max=self.data.domain.get_bounds()[0, 1])

        self.state = self.algorithm.initialize(rng_ask, self.evosax_param)

        x_init, y_init = self.data.get_n_best_output(self.hyperparameters.population).to_numpy()

        self.state = self.algorithm.tell(x_init, y_init.ravel(), self.state, self.evosax_param)

    def set_seed(self) -> None:
        ...

    def reset(self):
        self._check_imports()
        self.set_algorithm()

    def update_step(self, data_generator: DataGenerator) -> ExperimentData:
        _, rng_ask = jax.random.split(jax.random.PRNGKey(self.seed))

        # Ask for a set candidates
        x, state = self.algorithm.ask(rng_ask, self.state, self.evosax_param)

        # Evaluate the candidates
        x_experimentdata = ExperimentData.from_numpy(domain=self.domain, input_array=x)
        x_experimentdata.run(data_generator)

        _, y = x_experimentdata.to_numpy()

        # Update the strategy based on fitness
        self.state = self.algorithm.tell(x, y.ravel(), state, self.evosax_param)

        # return the data
        return ExperimentData.from_numpy(domain=self.domain,
                                         input_array=x,
                                         output_array=y)
