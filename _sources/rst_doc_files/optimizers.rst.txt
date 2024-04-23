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

======================== ========================================================================== =======================================================================================================
Name                     Keyword argument                                                           Reference
======================== ========================================================================== =======================================================================================================
CMAES                    ``"CMAES"``                                                                `pygmo cmaes <https://esa.github.io/pygmo2/algorithms.html#pygmo.cmaes>`_
PSO                      ``"PygmoPSO"``                                                             `pygmo pso_gen <https://esa.github.io/pygmo2/algorithms.html#pygmo.pso_gen>`_
SGA                      ``"SGA"``                                                                  `pygmo sga <https://esa.github.io/pygmo2/algorithms.html#pygmo.sga>`_
SEA                      ``"SEA"``                                                                  `pygmo sea <https://esa.github.io/pygmo2/algorithms.html#pygmo.sea>`_
XNES                     ``"XNES"``                                                                 `pygmo xnes <https://esa.github.io/pygmo2/algorithms.html#pygmo.xnes>`_
Differential Evolution   ``"DifferentialEvolution"``                                                `pygmo de <https://esa.github.io/pygmo2/algorithms.html#pygmo.de>`_
Simulated Annealing      ``"SimulatedAnnealing"``                                                   `pygmo simulated_annealing <https://esa.github.io/pygmo2/algorithms.html#pygmo.simulated_annealing>`_
======================== ========================================================================== =======================================================================================================

`Tensorflow keras`_ optimizers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These gradient based optimizers are ported from the `tensorflow keras <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_ Python library:

In order to use this optimizers you need to install the :code:`tensorflow` dependency:

.. code-block:: bash

    pip install tensorflow


======================== ====================================================================== =====================================================================================================
Name                     Keyword argument                                                       Reference
======================== ====================================================================== =====================================================================================================
SGD                      ``"SGD"``                                                              `tf.keras.optimizers.SGD <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD>`_
RMSprop                  ``"RMSprop"``                                                          `tf.keras.optimizers.RMSprop <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop>`_
Adam                     ``"AdamTensorflow"``                                                   `tf.keras.optimizers.Adam <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam>`_
Nadam                    ``"NAdam"``                                                            `tf.keras.optimizers.Nadam <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Nadam>`_
Adamax                   ``"Adamax"``                                                           `tf.keras.optimizers.Adamax <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adamax>`_
Ftrl                     ``"Ftrl"``                                                             `tf.keras.optimizers.Ftrl <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Ftrl>`_
======================== ====================================================================== =====================================================================================================

`Nevergrad`_ optimizers
^^^^^^^^^^^^^^^^^^^^^^^

These derivative-free global optimizers are ported from the `nevergrad <https://facebookresearch.github.io/nevergrad/index.html>`_ Python library:

In order to use this optimizers you need to install the :code:`nevergrad` dependency:

.. code-block:: bash

    pip install nevergrad

======================== ============================================================================================ =============================================================================================================================================================
Name                     Keyword argument                                                                             Reference
======================== ============================================================================================ =============================================================================================================================================================
Differential Evolution   ``"NevergradDE"``                                                                            `nevergrad.optimizers.DifferentialEvolution <https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.families.DifferentialEvolution>`_
PSO                      ``"PSO"``                                                                                    `nevergrad.optimizers.ConfPSO <https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.families.ConfPSO>`_
======================== ============================================================================================ =============================================================================================================================================================

`Evosax`_ optimizers
^^^^^^^^^^^^^^^^^^^^

These derivative-free global optimizers are ported from the `evosax <https://github.com/RobertTLange/evosax>`_ Python library:

In order to use this optimizers you need to install the :code:`evosax` dependency:

.. code-block:: bash

    pip install evosax

.. note::

    The `evosax <https://github.com/RobertTLange/evosax>`_ library is only available for Linux and Unix systems.

======================== ============================================================================================ =============================================================================================================================================================
Name                     Keyword argument                                                                             Reference
======================== ============================================================================================ =============================================================================================================================================================
CMAES                    ``"EvoSaxCMAES"``                                                                            `evosax.strategies.cma_es <https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/cma_es.py>`_
PSO                      ``"EvoSaxPSO"``                                                                              `evosax.strategies.pso <https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/pso.py>`_
Simulated Annealing      ``"EvoSaxSimAnneal"``                                                                        `evosax.strategies.sim_anneal <https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/sim_anneal.py>`_
Differential Evolution   ``"EvoSaxDE"``                                                                               `evosax.strategies.de <https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/de.py>`_
======================== ============================================================================================ =============================================================================================================================================================

`Optuna <https://optuna.readthedocs.io/en/stable/index.html>`_ optimizers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These derivative-free global optimizers are ported from the `optuna <https://optuna.readthedocs.io/en/stable/index.html>`_ Python library:

In order to use this optimizers you need to install the :code:`optuna` dependency:

.. code-block:: bash

    pip install optuna


================================ ========================================================================= ===========================================================================================================================================================================
Name                             Keyword argument                                                          Reference
================================ ========================================================================= ===========================================================================================================================================================================
Tree-structured Parzen Estimator ``"TPESampler"``                                                          `optuna.samplers.TPESampler <https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html#optuna.samplers.TPESampler>`_
================================ ========================================================================= ===========================================================================================================================================================================

`Optax <https://optax.readthedocs.io/en/latest/index.html>`_ optimizers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These derivative-free global optimizers are ported from the `optax <https://optax.readthedocs.io/en/latest/index.html>`_ Python library:

In order to use this optimizers you need to install the :code:`optax` dependency:

.. code-block:: bash

    pip install optax


================================ ========================================================================= ===========================================================================================================================================================================
Name                             Keyword argument                                                          Reference
================================ ========================================================================= ===========================================================================================================================================================================
Adam                             ``"Adam"``                                                                `optax.adam <https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adam>`_
SGD                              ``"SGDOptax"``                                                            `optax.sgd <https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.sgd>`_
================================ ========================================================================= ===========================================================================================================================================================================
