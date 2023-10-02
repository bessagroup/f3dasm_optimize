.. _Pygmo: https://esa.github.io/pygmo2/
.. _GPyOpt_link: https://sheffieldml.github.io/GPyOpt/
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
Name                     Keyword argument                                                           Reference
======================== ========================================================================== =======================================================================================================
CMAES                    ``"CMAES"``                                                                `pygmo cmaes <https://esa.github.io/pygmo2/algorithms.html#pygmo.cmaes>`_
PSO                      ``"PSO"``                                                                  `pygmo pso_gen <https://esa.github.io/pygmo2/algorithms.html#pygmo.pso_gen>`_
SGA                      ``"SGA"``                                                                  `pygmo sga <https://esa.github.io/pygmo2/algorithms.html#pygmo.sga>`_
SEA                      ``"SEA"``                                                                  `pygmo sea <https://esa.github.io/pygmo2/algorithms.html#pygmo.sea>`_
XNES                     ``"XNES"``                                                                 `pygmo xnes <https://esa.github.io/pygmo2/algorithms.html#pygmo.xnes>`_
Differential Evolution   ``"DifferentialEvolution"``                                                `pygmo de <https://esa.github.io/pygmo2/algorithms.html#pygmo.de>`_
Simulated Annealing      ``"SimulatedAnnealing"``                                                   `pygmo simulated_annealing <https://esa.github.io/pygmo2/algorithms.html#pygmo.simulated_annealing>`_
======================== ========================================================================== =======================================================================================================

`GPyOpt <https://sheffieldml.github.io/GPyOpt/>`_ implementations
^^^^^^^^^^^^^^^^^^^^^^^^^

======================== ========================================================================= ======================================================
Name                     Keyword argument                                                          Reference
======================== ========================================================================= ======================================================
Bayesian Optimization    ``"BayesianOptimization"``                                                `gpyopt <https://gpyopt.readthedocs.io/en/latest/>`_
======================== ========================================================================= ======================================================

`Tensorflow keras`_ optimizers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

======================== ====================================================================== =====================================================================================================
Name                     Keyword argument                                                       Reference
======================== ====================================================================== =====================================================================================================
SGD                      ``"SGD"``                                                              `tf.keras.optimizers.SGD <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD>`_
RMSprop                  ``"RMSprop"``                                                          `tf.keras.optimizers.RMSprop <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop>`_
Adam                     ``"Adam"``                                                             `tf.keras.optimizers.Adam <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam>`_
Nadam                    ``"NAdam"``                                                            `tf.keras.optimizers.Nadam <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Nadam>`_
Adamax                   ``"Adamax"``                                                           `tf.keras.optimizers.Adamax <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adamax>`_
Ftrl                     ``"Ftrl"``                                                             `tf.keras.optimizers.Ftrl <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Ftrl>`_
======================== ====================================================================== =====================================================================================================

`Nevergrad`_ optimizers
^^^^^^^^^^^^^^^^^^^^^^^

======================== ============================================================================================ =============================================================================================================================================================
Name                     Keyword argument                                                                             Reference
======================== ============================================================================================ =============================================================================================================================================================
Differential Evolution   ``"NevergradDE"``                                                                            `nevergrad.optimizers.DifferentialEvolution <https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.families.DifferentialEvolution>`_
PSO                      ``"NevergradPSO"``                                                                           `nevergrad.optimizers.ConfPSO <https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.families.ConfPSO>`_
======================== ============================================================================================ =============================================================================================================================================================

`Evosax`_ optimizers
^^^^^^^^^^^^^^^^^^^^

======================== ============================================================================================ =============================================================================================================================================================
Name                     Keyword argument                                                                             Reference
======================== ============================================================================================ =============================================================================================================================================================
CMAES                    ``"EvoSaxCMAES"``                                                                            `evosax.strategies.cma_es <https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/cma_es.py>`_
PSO                      ``"EvoSaxPSO"``                                                                              `evosax.strategies.pso <https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/pso.py>`_
Simulated Annealing      ``"EvoSaxSimAnneal"``                                                                        `evosax.strategies.sim_anneal <https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/sim_anneal.py>`_
Differential Evolution   ``"EvoSaxDE"``                                                                               `evosax.strategies.de <https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/de.py>`_
======================== ============================================================================================ =============================================================================================================================================================
