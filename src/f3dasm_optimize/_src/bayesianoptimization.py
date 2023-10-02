#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass
from typing import Any, List, Tuple

# Third-party
import GPy
import GPyOpt
import numpy as np

# Local
from ._protocol import DataGenerator
from .optimizer import Optimizer, OptimizerParameters

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


@dataclass
class BayesianOptimization_Parameters(OptimizerParameters):
    """Hyperparameters for BayesianOptimization optimizer"""

    model: Any = None
    space: Any = None
    acquisition: Any = None
    evaluator: Any = None
    de_duplication: Any = None


class BayesianOptimization(Optimizer):
    """Bayesian Optimization implementation from the GPyOPt library"""

    parameter: BayesianOptimization_Parameters = BayesianOptimization_Parameters()

    def init_parameters(self):
        domain = [
            {
                "name": f"{name}",
                "type": "continuous",
                "domain": (parameter.lower_bound, parameter.upper_bound),
            }
            for name, parameter in self.domain.get_continuous_input_parameters().items()
        ]

        kernel = GPy.kern.RBF(
            input_dim=len(self.domain))

        model = GPyOpt.models.gpmodel.GPModel(
            kernel=kernel,
            max_iters=1000,
            optimize_restarts=5,
            sparse=False,
            verbose=False,
        )

        space = GPyOpt.Design_space(space=domain)
        acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space)
        acquisition = GPyOpt.acquisitions.AcquisitionEI(
            model, space, acquisition_optimizer)
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

        # Default hyperparamaters
        options = {
            "model": model,
            "space": space,
            "acquisition": acquisition,
            "evaluator": evaluator,
            "de_duplication": True,
        }

        self.parameter = BayesianOptimization_Parameters(**options)

    def set_algorithm(self):
        self.algorithm = GPyOpt.methods.ModularBayesianOptimization(
            model=self.parameter.model,
            space=self.parameter.space,
            objective=None,
            acquisition=self.parameter.acquisition,
            evaluator=self.parameter.evaluator,
            X_init=None,
            Y_init=None,
            de_duplication=self.parameter.de_duplication,
        )

    def update_step(self, data_generator: DataGenerator) -> Tuple[np.ndarray, None]:

        self.algorithm.objective = GPyOpt.core.task.SingleObjective(
            data_generator.__call__)
        self.algorithm.X = self.data.input_data.to_numpy()
        self.algorithm.Y = self.data.output_data.to_numpy()

        x_new = self.algorithm.suggest_next_locations()

        return x_new, None

    def get_info(self) -> List[str]:
        return ['Stable', 'Robust', 'Global', 'Noisy', 'Derivative-Free', 'Single-Solution']