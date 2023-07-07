"""
Protocol classes from types outside the optimization submodule
"""
#                                                                       Modules
# =============================================================================

# Standard
try:
    from typing import Protocol
except ImportError:  # Python 3.7
    from typing_extensions import Protocol

# Third-party core
import numpy as np

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class DesignSpace(Protocol):
    """Protocol class for the designspace"""

    def get_continuous_input_parameters(self):  # List[ContinuousParameter]
        ...


class Function(Protocol):
    """Protocol class for the function"""

    def __call__(self) -> np.ndarray:
        """Evaluate the lossfunction"""
        ...

    def dfdx_legacy(x: np.ndarray) -> np.ndarray:
        """Retrieve the gradient. Legacy code!"""
        ...