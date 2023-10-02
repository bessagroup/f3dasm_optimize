.. _Pygmo: https://esa.github.io/pygmo2/
.. _GPyOpt: https://sheffieldml.github.io/GPyOpt/
.. _Tensorflow keras: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
.. _Nevergrad: https://facebookresearch.github.io/nevergrad/index.html
.. _EvoSax: https://github.com/RobertTLange/evosax

Implemented optimizers
======================

The following implementations of optimizers can found under this extension package: 

`Pygmo`_ implementations
^^^^^^^^^^^^^^^^^^^^^^^^

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

`GPyOpt`_ implementations
^^^^^^^^^^^^^^^^^^^^^^^^^

======================== ========================================================================= ======================================================
Name                      Docs of the Python class                                                 Reference
======================== ========================================================================= ======================================================
Bayesian Optimization    :class:`~f3dasm_optimize.bayesianoptimization.BayesianOptimization`       `gpyopt <https://gpyopt.readthedocs.io/en/latest/>`_
======================== ========================================================================= ======================================================

`Tensorflow keras`_ optimizers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

`Nevergrad`_ optimizers
^^^^^^^^^^^^^^^^^^^^^^^

======================== ============================================================================================ =============================================================================================================================================================
Name                      Docs of the Python class                                                                      Reference
======================== ============================================================================================ =============================================================================================================================================================
Differential Evolution   :class:`~f3dasm_optimize.differential_evoluation_nevergrad.DifferentialEvolution_Nevergrad`  `nevergrad.optimizers.DifferentialEvolution <https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.families.DifferentialEvolution>`_
PSO                      :class:`~f3dasm_optimize.pso_nevergrad.PSOConf`                                              `nevergrad.optimizers.ConfPSO <https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.families.ConfPSO>`_
======================== ============================================================================================ =============================================================================================================================================================

`Evosax`_ optimizers
^^^^^^^^^^^^^^^^^^^^

======================== ============================================================================================ =============================================================================================================================================================
Name                      Docs of the Python class                                                                      Reference
======================== ============================================================================================ =============================================================================================================================================================
CMAES                    :class:`~f3dasm_optimize.evosax_implementations.EvoSaxCMAES`                                 `evosax.strategies.cma_es <https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/cma_es.py>`_
PSO                      :class:`~f3dasm_optimize.evosax_implementations.EvoSaxPSO`                                   `evosax.strategies.pso <https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/pso.py>`_
Simulated Annealing      :class:`~f3dasm_optimize.evosax_implementations.EvoSaxSimAnneal`                             `evosax.strategies.sim_anneal <https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/sim_anneal.py>`_
Differential Evolution   :class:`~f3dasm_optimize.evosax_implementations.EvoSaxDE`                                    `evosax.strategies.de <https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/de.py>`_

======================== ============================================================================================ =============================================================================================================================================================
