Implemented optimizers
======================

The following implementations of optimizers can found under this extension package: 

These are ported from several libraries such as `GPyOpt <https://sheffieldml.github.io/GPyOpt/>`_, `tensorflow <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_ and `pygmo <https://esa.github.io/pygmo2/>`_.


Pygmo implementations
^^^^^^^^^^^^^^^^^^^^^

These derivative-free global optimizers are ported from the `pygmo <https://esa.github.io/pygmo2/>`_ Python library: 

======================== ========================================================================== =======================================================================================================
Name                      Docs of the Python class                                                  Reference
======================== ========================================================================== =======================================================================================================
CMAES                    :class:`~f3dasm_optimize.cmaes.CMAES`                                      `pygmo cmaes <https://esa.github.io/pygmo2/algorithms.html#pygmo.cmaes>`_
PSO                      :class:`~f3dasm_optimize.pso.PSO`                                          `pygmo pso_gen <https://esa.github.io/pygmo2/algorithms.html#pygmo.pso_gen>`_
SGA                      :class:`~f3dasm_optimize.sga.SGA`                                          `pygmo sga <https://esa.github.io/pygmo2/algorithms.html#pygmo.sga>`_
SEA                      :class:`~f3dasm_optimize.sea.SEA`                                          `pygmo sea <https://esa.github.io/pygmo2/algorithms.html#pygmo.sea>`_
XNES                     :class:`~f3dasm_optimize.xnes.XNES`                                        `pygmo xnes <https://esa.github.io/pygmo2/algorithms.html#pygmo.xnes>`_
Differential Evolution   :class:`~f3dasm_optimize.differentialevoluation.DifferentialEvolution`     `pygmo de <https://esa.github.io/pygmo2/algorithms.html#pygmo.de>`_
Simulated Annealing      :class:`~f3dasm_optimize.simulatedannealing.SimulatedAnnealing`            `pygmo simulated_annealing <https://esa.github.io/pygmo2/algorithms.html#pygmo.simulated_annealing>`_
======================== ========================================================================== =======================================================================================================

GPyOpt Implementations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

======================== ========================================================================= ======================================================
Name                      Docs of the Python class                                                 Reference
======================== ========================================================================= ======================================================
Bayesian Optimization    :class:`~f3dasm_optimize.bayesianoptimization.BayesianOptimization`       `GPyOpt <https://gpyopt.readthedocs.io/en/latest/>`_
======================== ========================================================================= ======================================================

Tensorflow Keras optimizers
^^^^^^^^^^^^^^^^^^^^^^^^^^^

======================== ====================================================================== =====================================================================================================
Name                      Docs of the Python class                                              Reference
======================== ====================================================================== =====================================================================================================
SGD                      :class:`~f3dasm_optimize.sgd.SGD`                                      `tf.keras.optimizers.SGD <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD>`_
RMSprop                  :class:`~f3dasm_optimize.rmsprop.RMSprop`                              `tf.keras.optimizers.RMSprop <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop>`_
Adam                     :class:`~f3dasm_optimize.adam.Adam`                                    `tf.keras.optimizers.Adam <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam>`_
Nadam                    :class:`~f3dasm_optimize.nadam.Nadam`                                  `tf.keras.optimizers.Nadam <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Nadam>`_
Adamax                   :class:`~f3dasm_optimize.adamax.Adamax`                                `tf.keras.optimizers.Adamax <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adamax>`_
Ftrl                     :class:`~f3dasm_optimize.ftrl.Ftrl`                                    `tf.keras.optimizers.Ftrl <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Ftrl>`_
======================== ====================================================================== =====================================================================================================
