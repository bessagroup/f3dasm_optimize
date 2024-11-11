.. _Pygmo: https://esa.github.io/pygmo2/
.. _Tensorflow keras: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
.. _Nevergrad: https://facebookresearch.github.io/nevergrad/index.html
.. _EvoSax: https://github.com/RobertTLange/evosax


Implemented optimizers
======================

The following implementations of optimizers can found under this extension package: 

`Pygmo`_ implementations
^^^^^^^^^^^^^^^^^^^^^^^^

These derivative-free global optimizers are ported from the `pygmo <https://esa.github.io/pygmo2/>`_ Python library: 

In order to use this optimizers you need to install the :code:`pygmo` dependency:

.. code-block:: bash

    pip install pygmo

.. note::

    The `pygmo <https://esa.github.io/pygmo2/>`_ library is not compatible with Python 3.9 yet and only available for Linux and Unix systems.

======================== ========================================================================== ================================================== =======================================================================================================
Name                     Keyword argument                                                           Function                                           Reference
======================== ========================================================================== ================================================== =======================================================================================================
CMAES                    ``"CMAES"``                                                                :func:`~f3dasm_optimize.cmaes_pygmo`               `pygmo cmaes <https://esa.github.io/pygmo2/algorithms.html#pygmo.cmaes>`_
PSO                      ``"PygmoPSO"``                                                             :func:`~f3dasm_optimize.pso_pygmo`                 `pygmo pso_gen <https://esa.github.io/pygmo2/algorithms.html#pygmo.pso_gen>`_
SGA                      ``"SGA"``                                                                  :func:`~f3dasm_optimize.sga`                       `pygmo sga <https://esa.github.io/pygmo2/algorithms.html#pygmo.sga>`_
SEA                      ``"SEA"``                                                                  :func:`~f3dasm_optimize.sea`                       `pygmo sea <https://esa.github.io/pygmo2/algorithms.html#pygmo.sea>`_
XNES                     ``"XNES"``                                                                 :func:`~f3dasm_optimize.xnes`                      `pygmo xnes <https://esa.github.io/pygmo2/algorithms.html#pygmo.xnes>`_
Differential Evolution   ``"DifferentialEvolution"``                                                :func:`~f3dasm_optimize.de_pygmo`                  `pygmo de <https://esa.github.io/pygmo2/algorithms.html#pygmo.de>`_
Simulated Annealing      ``"SimulatedAnnealing"``                                                   :func:`~f3dasm_optimize.simanneal_pygmo`           `pygmo simulated_annealing <https://esa.github.io/pygmo2/algorithms.html#pygmo.simulated_annealing>`_
======================== ========================================================================== ================================================== =======================================================================================================


`Tensorflow keras`_ optimizers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These gradient based optimizers are ported from the `tensorflow keras <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_ Python library:

In order to use this optimizers you need to install the :code:`tensorflow` dependency:

.. code-block:: bash

    pip install tensorflow


======================== ====================================================================== ============================================ =====================================================================================================
Name                     Keyword argument                                                       Function                                     Reference
======================== ====================================================================== ============================================ =====================================================================================================
SGD                      ``"SGD"``                                                              :func:`~f3dasm_optimize.sgd_tensorflow`      `tf.keras.optimizers.SGD <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD>`_
RMSprop                  ``"RMSprop"``                                                          :func:`~f3dasm_optimize.rmsprop`             `tf.keras.optimizers.RMSprop <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop>`_
Adam                     ``"AdamTensorflow"``                                                   :func:`~f3dasm_optimize.adam_tensorflow`     `tf.keras.optimizers.Adam <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam>`_
Nadam                    ``"NAdam"``                                                            :func:`~f3dasm_optimize.nadam`               `tf.keras.optimizers.Nadam <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Nadam>`_
Adamax                   ``"Adamax"``                                                           :func:`~f3dasm_optimize.adamax`              `tf.keras.optimizers.Adamax <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adamax>`_
Ftrl                     ``"Ftrl"``                                                             :func:`~f3dasm_optimize.ftrl`                `tf.keras.optimizers.Ftrl <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Ftrl>`_
======================== ====================================================================== ============================================ =====================================================================================================




`Nevergrad`_ optimizers
^^^^^^^^^^^^^^^^^^^^^^^

These derivative-free global optimizers are ported from the `nevergrad <https://facebookresearch.github.io/nevergrad/index.html>`_ Python library:

In order to use this optimizers you need to install the :code:`nevergrad` dependency:

.. code-block:: bash

    pip install nevergrad

======================== ============================================================================================ ============================================= =============================================================================================================================================================
Name                     Keyword argument                                                                             Function                                      Reference
======================== ============================================================================================ ============================================= =============================================================================================================================================================
Differential Evolution   ``"NevergradDE"``                                                                            :func:`~f3dasm_optimize.de_nevergrad`         `nevergrad.optimizers.DifferentialEvolution <https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.families.DifferentialEvolution>`_
PSO                      ``"PSO"``                                                                                    :func:`~f3dasm_optimize.pso_nevergrad`        `nevergrad.optimizers.ConfPSO <https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.families.ConfPSO>`_
======================== ============================================================================================ ============================================= =============================================================================================================================================================


`Evosax`_ optimizers
^^^^^^^^^^^^^^^^^^^^

These derivative-free global optimizers are ported from the `evosax <https://github.com/RobertTLange/evosax>`_ Python library:

In order to use this optimizers you need to install the :code:`evosax` dependency:

.. code-block:: bash

    pip install evosax

.. note::

    The `evosax <https://github.com/RobertTLange/evosax>`_ library is only available for Linux and Unix systems.

======================== ============================================================================================ ============================================= =============================================================================================================================================================
Name                     Keyword argument                                                                             Function                                      Reference
======================== ============================================================================================ ============================================= =============================================================================================================================================================
CMAES                    ``"EvoSaxCMAES"``                                                                            :func:`~f3dasm_optimize.cmaes`                `evosax.strategies.cma_es <https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/cma_es.py>`_
PSO                      ``"EvoSaxPSO"``                                                                              :func:`~f3dasm_optimize.pso`                  `evosax.strategies.pso <https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/pso.py>`_
Simulated Annealing      ``"EvoSaxSimAnneal"``                                                                        :func:`~f3dasm_optimize.simanneal`            `evosax.strategies.sim_anneal <https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/sim_anneal.py>`_
Differential Evolution   ``"EvoSaxDE"``                                                                               :func:`~f3dasm_optimize.de`                   `evosax.strategies.de <https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/de.py>`_
======================== ============================================================================================ ============================================= =============================================================================================================================================================


`Optuna <https://optuna.readthedocs.io/en/stable/index.html>`_ optimizers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These derivative-free global optimizers are ported from the `optuna <https://optuna.readthedocs.io/en/stable/index.html>`_ Python library:

In order to use this optimizers you need to install the :code:`optuna` dependency:

.. code-block:: bash

    pip install optuna

================================ ========================================================================= ============================================= ===========================================================================================================================================================================
Name                             Keyword argument                                                          Function                                      Reference
================================ ========================================================================= ============================================= ===========================================================================================================================================================================
Tree-structured Parzen Estimator ``"TPESampler"``                                                          :func:`~f3dasm_optimize.tpe_sampler`              `optuna.samplers.TPESampler <https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html#optuna.samplers.TPESampler>`_
================================ ========================================================================= ============================================= ===========================================================================================================================================================================


`Optax <https://optax.readthedocs.io/en/latest/index.html>`_ optimizers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These derivative-free global optimizers are ported from the `optax <https://optax.readthedocs.io/en/latest/index.html>`_ Python library:

In order to use this optimizers you need to install the :code:`optax` dependency:

.. code-block:: bash

    pip install optax


================================ ========================================================================= ============================================= ===========================================================================================================================================================================
Name                             Keyword argument                                                          Function                                      Reference
================================ ========================================================================= ============================================= ===========================================================================================================================================================================
Adam                             ``"Adam"``                                                                :func:`~f3dasm_optimize.adam`                 `optax.adam <https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adam>`_
SGD                              ``"SGDOptax"``                                                            :func:`~f3dasm_optimize.sgd`                  `optax.sgd <https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.sgd>`_
================================ ========================================================================= ============================================= ===========================================================================================================================================================================
