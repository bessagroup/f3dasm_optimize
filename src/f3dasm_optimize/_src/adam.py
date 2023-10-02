"""
Information on the Adam optimizer
"""
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
class Adam_Parameters(OptimizerParameters):
    """Hyperparameters for Adam optimizer"""

    learning_rate: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-07
    amsgrad: bool = False


class Adam(TensorflowOptimizer):
    """Adam"""

    hyperparameters: Adam_Parameters = Adam_Parameters()

    def set_algorithm(self):
        self.algorithm = tf.keras.optimizers.Adam(
            learning_rate=self.hyperparameters.learning_rate,
            beta_1=self.hyperparameters.beta_1,
            beta_2=self.hyperparameters.beta_2,
            epsilon=self.hyperparameters.epsilon,
            amsgrad=self.hyperparameters.amsgrad,
        )

    def get_info(self) -> List[str]:
        return ['Stable', 'Global', 'First-Order', 'Single-Solution']
