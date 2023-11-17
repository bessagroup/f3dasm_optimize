Getting Started
===============

In order to use this library, you need to have a working installation of :code:`f3dasm`.
You can get the installation instructions from `here <https://f3dasm.readthedocs.io/en/latest/>`_.

Then, you can install this optimization extension library using pip:

.. code-block:: bash

    pip install f3dasm_optimize

Upon installing the library in your environment, when accessing the :code:`f3dasm` library, the optimization
extension will be automatically loaded when you look at the available optimizers:

.. code-block:: python

    import f3dasm

    print(f3dasm.optimization.OPTIMIZERS)
    >>> ['RandomSearch', 'CG', 'LBFGSB', 'NelderMead']

Wait a second .. these are only the `optimizers that can be found in the 
standard f3dasm library <https://f3dasm.readthedocs.io/en/latest/rst_doc_files/classes/optimization/optimizers.html#implemented-optimizers>`_!
What happened to the new optimizers?

Well, the new optimizers are not loaded by default, because you might not have the required
dependencies installed. In order to load the new optimizers, you need to explicitly have the
dependencies for the optimizers installed in your environment.

For example, if you want to use the `TPESampler <https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html#optuna.samplers.TPESampler>`_ 
optimizer from `optuna <https://optuna.readthedocs.io/en/stable/index.html>`_, you need to have optuna installed in your environment:

.. code-block:: bash

    pip install optuna

Then, when you inspect the optimizers, the list will be updated:

.. code-block:: python

    import f3dasm

    print(f3dasm.optimization.OPTIMIZERS)
    >>> ['RandomSearch', 'CG', 'LBFGSB', 'NelderMead', 'TPESampler']

Now, you can use the new optimizers in your code:

.. code-block:: python

    experimentdata.optimize(optimizer='TPESampler', data_generator='Ackley', iterations=100)

.. note::

    Some optimizers are not compatible with particular versions of Python or operating systems.
    Consult the documentation of the optimizer you want to use to see if there are any other requirements
    that need to be met.

Happy optimizing!