#                                                                       Modules
# =============================================================================

# Standard

from ._imports import try_import

with try_import() as _evosax_imports:
    from .evosax_optimizers import cmaes, de, pso, simanneal

with try_import() as _nevergrad_imports:
    from .nevergrad_optimizers import de_nevergrad, pso_nevergrad

with try_import() as _pygmo_imports:
    from .pygmo_optimizers import (cmaes_pygmo, de_pygmo, pso_pygmo, sade, sea,
                                   sga, simanneal_pygmo, xnes)

with try_import() as _optuna_imports:
    from .optuna_optimizers import tpe_sampler

with try_import() as _tensorflow_imports:
    from .tensorflow_optimizers import (adam_tensorflow, adamax, ftrl, nadam,
                                        rmsprop, sgd_tensorflow)

with try_import() as _optax_imports:
    from .optax_optimizers import adam, sgd

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


def optimizers_extension():
    optimizer_list = []

    if _pygmo_imports.is_successful():
        optimizer_list.extend([cmaes_pygmo, de_pygmo, pso_pygmo, sade, sea,
                               sga, simanneal_pygmo, xnes])

    if _optuna_imports.is_successful():
        optimizer_list.extend([tpe_sampler])

    if _tensorflow_imports.is_successful():
        optimizer_list.extend([adam_tensorflow, adamax, ftrl, nadam,
                               rmsprop, sgd_tensorflow])

    if _evosax_imports.is_successful():
        optimizer_list.extend([cmaes, de, pso, simanneal])

    if _nevergrad_imports.is_successful():
        optimizer_list.extend([de_nevergrad, pso_nevergrad])

    if _optax_imports.is_successful():
        optimizer_list.extend([adam, sgd])

    return optimizer_list


__all__ = [
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
    'sgd',
    'sgd_tensorflow',
    'simanneal',
    'simanneal_pygmo',
    'tpe_sampler',
    'xnes',
]
