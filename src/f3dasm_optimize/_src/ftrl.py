#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass
from typing import List

# Locals
from f3dasm import try_import
from f3dasm.optimization import OptimizerParameters

from .adapters.tensorflow_implementations import TensorflowOptimizer

# Third-party extension
with try_import('optimization') as _imports:
    import tensorflow as tf

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


@dataclass
class Ftrl_Parameters(OptimizerParameters):
    """Hyperparameters for Ftrl optimizer"""

    learning_rate: float = 0.001
    learning_rate_power: float = -0.5
    initial_accumulator_value: float = 0.1
    l1_regularization_strength: float = 0.0
    l2_regularization_strength: float = 0.0
    l2_shrinkage_regularization_strength: float = 0.0
    beta: float = 0.0


class Ftrl(TensorflowOptimizer):
    """Ftrl"""

    hyperparameters: Ftrl_Parameters = Ftrl_Parameters()

    def set_algorithm(self):
        self.algorithm = tf.keras.optimizers.Ftrl(
            learning_rate=self.hyperparameters.learning_rate,
            learning_rate_power=self.hyperparameters.learning_rate_power,
            initial_accumulator_value=self.hyperparameters.initial_accumulator_value,
            l1_regularization_strength=self.hyperparameters.l1_regularization_strength,
            l2_regularization_strength=self.hyperparameters.l2_regularization_strength,
            l2_shrinkage_regularization_strength=self.hyperparameters.l2_shrinkage_regularization_strength,
            beta=self.hyperparameters.beta,
        )

    def get_info(self) -> List[str]:
        return ['Fast', 'Single-Solution']
