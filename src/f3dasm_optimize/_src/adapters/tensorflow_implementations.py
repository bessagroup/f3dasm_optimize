#                                                                       Modules
# =============================================================================

# Standard
from typing import Callable, Optional, Tuple

# Third-party
import autograd.core
import autograd.numpy as np
import tensorflow as tf
from autograd import elementwise_grad as egrad
from f3dasm import ExperimentData
from f3dasm.datageneration import DataGenerator
from f3dasm.optimization import Optimizer
from keras import Model

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling', 'Surya Manoj Sanu']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class TensorflowOptimizer(Optimizer):
    require_gradients: bool = True

    def __init__(self, algorithm_cls, seed: Optional[int] = None,
                 **hyperparameters):
        self.algorithm_cls = algorithm_cls
        self.seed = seed
        self.hyperparameters = hyperparameters

    def init(self, data: ExperimentData, data_generator: DataGenerator):
        self.data = data
        self.data_generator = data_generator
        self.algorithm = self.algorithm_cls(**self.hyperparameters)

        self.args = {}

        def fitness(x: np.ndarray) -> np.ndarray:
            evaluated_sample = self.data_generator._run(
                x, domain=self.data.domain)

            _, y_ = evaluated_sample.to_numpy()
            return y_

        self.args["model"] = _SimpelModel(
            None,
            args={
                "dim": len(self.data.domain),
                "x0": self.data.get_n_best_output(1).to_numpy()[0],
                "bounds": self.data.domain.get_bounds(),
            },
        )  # Build the model
        self.args["tvars"] = self.args["model"].trainable_variables

        # TODO: This is an important conversion!!
        self.args["func"] = _convert_autograd_to_tensorflow(
            self.data_generator.__call__)

    def update_step(self) -> Tuple[np.ndarray, np.ndarray]:
        with tf.GradientTape() as tape:
            tape.watch(self.args["tvars"])
            # tf.float32
            logits = 0.0 + tf.cast(self.args["model"](None), tf.float64)
            loss = self.args["func"](tf.reshape(
                logits, (len(self.data.domain))))

        grads = tape.gradient(loss, self.args["tvars"])
        self.algorithm.apply_gradients(zip(grads, self.args["tvars"]))

        logits = 0.0 + tf.cast(self.args["model"](None), tf.float64)
        loss = self.args["func"](tf.reshape(
            logits, (len(self.data.domain))))

        x = logits.numpy().copy()
        y = loss.numpy().copy()

        # return the data
        return x, np.atleast_2d(np.array(y))


def _convert_autograd_to_tensorflow(func: Callable):
    """Convert autograd function to tensorflow function

    Parameters
    ----------
    func
        callable function to convert

    Returns
    -------
        wrapper to convert autograd function to tensorflow function
    """
    @tf.custom_gradient
    def wrapper(x, *args, **kwargs):
        vjp, ans = autograd.core.make_vjp(func, x.numpy())

        def first_grad(dy):
            @tf.custom_gradient
            def jacobian(a):
                vjp2, ans2 = autograd.core.make_vjp(egrad(func), a.numpy())
                return ans2, vjp2  # hessian

            return dy * jacobian(x)

        return ans, first_grad

    return wrapper


class _Model(Model):
    def __init__(self, seed=None, args=None):
        super().__init__()
        self.seed = seed
        self.env = args


class _SimpelModel(_Model):
    """
    The class for performing optimization in the input space of the functions.
    """

    def __init__(self, seed=None, args=None):
        super().__init__(seed)
        self.z = tf.Variable(
            args["x0"],
            trainable=True,
            dtype=tf.float32,
            constraint=lambda x: tf.clip_by_value(
                x,
                clip_value_min=args["bounds"][:, 0],
                clip_value_max=args["bounds"][:, 1],
            ),
        )  # S:ADDED

    def call(self, inputs=None):
        return self.z
