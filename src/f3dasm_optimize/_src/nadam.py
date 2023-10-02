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
class Nadam_Parameters(OptimizerParameters):
    """Hyperparameters for Momentum optimizer
    )
    """

    learning_rate: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-07


class Nadam(TensorflowOptimizer):
    """Nadam"""

    hyperparameters: Nadam_Parameters = Nadam_Parameters()

    def set_algorithm(self):
        self.algorithm = tf.keras.optimizers.Nadam(
            learning_rate=self.hyperparameters.learning_rate,
            beta_1=self.hyperparameters.beta_1,
            beta_2=self.hyperparameters.beta_2,
            epsilon=self.hyperparameters.epsilon,
        )

    def get_info(self) -> List[str]:
        return ['Stable', 'Global', 'First-Order', 'Single-Solution']
