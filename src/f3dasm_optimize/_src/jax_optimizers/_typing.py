#                                                                       Modules
# =============================================================================

# Standard
from typing import Callable, Dict, List, Tuple

# Third-party
from jax import Array
from jaxtyping import PyTree
from optax import OptState

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


ScanFunction = Callable[
    [Tuple[PyTree, OptState, Array, Dict[str, Array]]],
    Tuple[Tuple[PyTree, OptState, Array],
          Dict[str, Array]]]

InitFunction = Callable[[], OptState]

LossFunction = Callable[[PyTree], Array]

PostFunction = Callable[[PyTree],
                        Tuple[List[Dict[str, PyTree]],
                              List[Dict[str, PyTree]]]]
