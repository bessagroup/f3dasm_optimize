"""
Optimizers for PyTrees using JAX.
"""

#                                                                       Modules
# =============================================================================

# Local
from f3dasm_optimize._src._imports import try_import
from f3dasm_optimize._src.jax_optimizers import (EvoSaxUpdate, UpdateStep,
                                                 evosax_scan, lbfgs_scan,
                                                 optax_scan)

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

__all__ = [
    'try_import',
    'EvoSaxUpdate',
    'UpdateStep',
    'evosax_scan',
    'lbfgs_scan',
    'optax_scan',
]
