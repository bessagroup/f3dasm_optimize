.. _Nevergrad: https://facebookresearch.github.io/nevergrad/index.html
.. _EvoSax: https://github.com/RobertTLange/evosax


Implemented optimizers
======================

The following implementations of optimizers can found under this extension package: 


`Nevergrad`_ optimizers
^^^^^^^^^^^^^^^^^^^^^^^

These derivative-free global optimizers are ported from the `nevergrad <https://facebookresearch.github.io/nevergrad/index.html>`_ Python library:

In order to use this optimizers you need to install the :code:`nevergrad` dependency:

.. code-block:: bash

    pip install nevergrad

======================== ============================================================================================ ============================================= =============================================================================================================================================================
Name                     Keyword argument                                                                             Function                                      Reference
======================== ============================================================================================ ============================================= =============================================================================================================================================================
Differential Evolution   ``"de_nevergrad"``                                                                           :func:`~f3dasm_optimize.de_nevergrad`         `nevergrad.optimizers.DifferentialEvolution <https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.families.DifferentialEvolution>`_
PSO                      ``"pso_nevergrad"``                                                                          :func:`~f3dasm_optimize.pso_nevergrad`        `nevergrad.optimizers.ConfPSO <https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.families.ConfPSO>`_
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
CMAES                    ``"cmaes"``                                                                                  :func:`~f3dasm_optimize.cmaes`                `evosax.strategies.cma_es <https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/cma_es.py>`_
PSO                      ``"pso"``                                                                                    :func:`~f3dasm_optimize.pso`                  `evosax.strategies.pso <https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/pso.py>`_
Simulated Annealing      ``"simanneal"``                                                                              :func:`~f3dasm_optimize.simanneal`            `evosax.strategies.sim_anneal <https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/sim_anneal.py>`_
Differential Evolution   ``"de"``                                                                                     :func:`~f3dasm_optimize.de`                   `evosax.strategies.de <https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/de.py>`_
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
Tree-structured Parzen Estimator ``"tpe_sampler"``                                                         :func:`~f3dasm_optimize.tpe_sampler`          `optuna.samplers.TPESampler <https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html#optuna.samplers.TPESampler>`_
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
Adam                             ``"adam"``                                                                :func:`~f3dasm_optimize.adam`                 `optax.adam <https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adam>`_
SGD                              ``"sgd"``                                                                 :func:`~f3dasm_optimize.sgd`                  `optax.sgd <https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.sgd>`_
================================ ========================================================================= ============================================= ===========================================================================================================================================================================
