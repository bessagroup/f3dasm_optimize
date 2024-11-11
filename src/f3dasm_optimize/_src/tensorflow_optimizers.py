"""
Information on the Adam optimizer
"""
#                                                                       Modules
# =============================================================================

# Third-party
import tensorflow as tf

# Locals
from ._protocol import OptimizerTuple
from .adapters.tensorflow_implementations import TensorflowOptimizer

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


def adam_tensorflow(learning_rate: float = 0.001, beta_1: float = 0.9,
                    beta_2: float = 0.999, epsilon: float = 1e-07,
                    amsgrad: bool = False, **kwargs) -> OptimizerTuple:
    """
    Adam optimizer using TensorFlow.

    Parameters
    ----------
    learning_rate : float, optional
        Learning rate, by default 0.001.
    beta_1 : float, optional
        Exponential decay rate for the first moment, by default 0.9.
    beta_2 : float, optional
        Exponential decay rate for the second moment, by default 0.999.
    epsilon : float, optional
        Small constant for numerical stability, by default 1e-07.
    amsgrad : bool, optional
        Whether to apply the AMSGrad variant, by default False.

    Returns
    -------
    OptimizerTuple
        OptimizerTuple object.
    """
    return OptimizerTuple(
        base_class=TensorflowOptimizer,
        algorithm=tf.keras.optimizers.Adam,
        hyperparameters={
            'learning_rate': learning_rate,
            'beta_1': beta_1,
            'beta_2': beta_2,
            'epsilon': epsilon,
            'amsgrad': amsgrad,
            **kwargs
        }
    )


# =============================================================================

def adamax(learning_rate: float = 0.001, beta_1: float = 0.9,
           beta_2: float = 0.999, epsilon: float = 1e-07,
           **kwargs) -> OptimizerTuple:
    """
    Adamax optimizer using TensorFlow.

    Parameters
    ----------
    learning_rate : float, optional
        Learning rate, by default 0.001.
    beta_1 : float, optional
        Exponential decay rate for the first moment, by default 0.9.
    beta_2 : float, optional
        Exponential decay rate for the second moment, by default 0.999.
    epsilon : float, optional
        Small constant for numerical stability, by default 1e-07.

    Returns
    -------
    OptimizerTuple
        OptimizerTuple object.
    """
    return OptimizerTuple(
        base_class=TensorflowOptimizer,
        algorithm=tf.keras.optimizers.Adamax,
        hyperparameters={
            'learning_rate': learning_rate,
            'beta_1': beta_1,
            'beta_2': beta_2,
            'epsilon': epsilon,
            **kwargs
        }
    )


# =============================================================================

def ftrl(learning_rate: float = 0.001, learning_rate_power: float = -0.5,
         initial_accumulator_value: float = 0.1,
         l1_regularization_strength: float = 0.0,
         l2_regularization_strength: float = 0.0,
         l2_shrinkage_regularization_strength: float = 0.0,
         beta: float = 0.0, **kwargs) -> OptimizerTuple:
    """
    Ftrl optimizer using TensorFlow.

    Parameters
    ----------
    learning_rate : float, optional
        Learning rate, by default 0.001.
    learning_rate_power : float, optional
        Power to scale the learning rate, by default -0.5.
    initial_accumulator_value : float, optional
        Initial value for accumulators, by default 0.1.
    l1_regularization_strength : float, optional
        Strength of L1 regularization, by default 0.0.
    l2_regularization_strength : float, optional
        Strength of L2 regularization, by default 0.0.
    l2_shrinkage_regularization_strength : float, optional
        Strength of L2 shrinkage regularization, by default 0.0.
    beta : float, optional
        Beta parameter, by default 0.0.

    Returns
    -------
    OptimizerTuple
        OptimizerTuple object.
    """
    return OptimizerTuple(
        base_class=TensorflowOptimizer,
        algorithm=tf.keras.optimizers.Ftrl,
        hyperparameters={
            'learning_rate': learning_rate,
            'learning_rate_power': learning_rate_power,
            'initial_accumulator_value': initial_accumulator_value,
            'l1_regularization_strength': l1_regularization_strength,
            'l2_regularization_strength': l2_regularization_strength,
            'l2_shrinkage_regularization_strength':
            l2_shrinkage_regularization_strength,
            'beta': beta,
            **kwargs
        }
    )


# =============================================================================


def nadam(learning_rate: float = 0.001, beta_1: float = 0.9,
          beta_2: float = 0.999, epsilon: float = 1e-07,
          **kwargs) -> OptimizerTuple:
    """
    Nadam optimizer using TensorFlow.

    Parameters
    ----------
    learning_rate : float, optional
        Learning rate, by default 0.001.
    beta_1 : float, optional
        Exponential decay rate for the first moment, by default 0.9.
    beta_2 : float, optional
        Exponential decay rate for the second moment, by default 0.999.
    epsilon : float, optional
        Small constant for numerical stability, by default 1e-07.

    Returns
    -------
    OptimizerTuple
        OptimizerTuple object.
    """
    return OptimizerTuple(
        base_class=TensorflowOptimizer,
        algorithm=tf.keras.optimizers.Nadam,
        hyperparameters={
            'learning_rate': learning_rate,
            'beta_1': beta_1,
            'beta_2': beta_2,
            'epsilon': epsilon,
            **kwargs
        }
    )


# =============================================================================


def rmsprop(learning_rate: float = 0.001, rho: float = 0.9,
            momentum: float = 0.0, epsilon: float = 1e-07,
            centered: bool = False, **kwargs) -> OptimizerTuple:
    """
    RMSprop optimizer using TensorFlow.

    Parameters
    ----------
    learning_rate : float, optional
        Learning rate, by default 0.001.
    rho : float, optional
        Decay rate, by default 0.9.
    momentum : float, optional
        Momentum value, by default 0.0.
    epsilon : float, optional
        Small constant for numerical stability, by default 1e-07.
    centered : bool, optional
        Whether to use centered version, by default False.

    Returns
    -------
    OptimizerTuple
        OptimizerTuple object.
    """
    return OptimizerTuple(
        base_class=TensorflowOptimizer,
        algorithm=tf.keras.optimizers.RMSprop,
        hyperparameters={
            'learning_rate': learning_rate,
            'rho': rho,
            'momentum': momentum,
            'epsilon': epsilon,
            'centered': centered,
            **kwargs
        }
    )


# =============================================================================
def sgd_tensorflow(learning_rate: float = 0.01, momentum: float = 0.0,
                   nesterov: bool = False, **kwargs) -> OptimizerTuple:
    """
    SGD optimizer using TensorFlow.

    Parameters
    ----------
    learning_rate : float, optional
        Learning rate, by default 0.01.
    momentum : float, optional
        Momentum value, by default 0.0.
    nesterov : bool, optional
        Whether to use Nesterov momentum, by default False.

    Returns
    -------
    OptimizerTuple
        OptimizerTuple object.
    """
    return OptimizerTuple(
        base_class=TensorflowOptimizer,
        algorithm=tf.keras.optimizers.SGD,
        hyperparameters={
            'learning_rate': learning_rate,
            'momentum': momentum,
            'nesterov': nesterov,
            **kwargs
        }
    )


# =============================================================================
