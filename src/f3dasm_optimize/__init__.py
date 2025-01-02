#                                                                       Modules
# =============================================================================

# Local
from ._src import optimizers_extension
from ._src._imports import try_import

with try_import() as _evosax_imports:
    from ._src.evosax_optimizers import cmaes, de, pso, simanneal

with try_import() as _nevergrad_imports:
    from ._src.nevergrad_optimizers import de_nevergrad, pso_nevergrad

with try_import() as _pygmo_imports:
    from ._src.pygmo_optimizers import (cmaes_pygmo, de_pygmo, pso_pygmo, sade,
                                        sea, sga, simanneal_pygmo, xnes)

with try_import() as _optuna_imports:
    from ._src.optuna_optimizers import tpe_sampler

with try_import() as _tensorflow_imports:
    from ._src.tensorflow_optimizers import (adam_tensorflow, adamax, ftrl,
                                             nadam, rmsprop, sgd_tensorflow)

with try_import() as _optax_imports:
    from ._src.optax_optimizers import adam, sgd

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

__all__ = [
    'optimizers_extension',
    'adam',
    'adam_tensorflow',
    'adamax',
    'cmaes',
    'cmaes_pygmo',
    'de',
    'de_nevergrad',
    'de_pygmo',
    'ftrl',
    'nadam',
    'pso',
    'pso_nevergrad',
    'pso_pygmo',
    'rmsprop',
    'sade',
    'sea',
    'sga',
    'sgd_tensorflow',
    'sgd',
    'simanneal',
    'simanneal_pygmo',
    'tpe_sampler',
    'xnes',
]

__version__ = '1.5.5'
