#                                                                       Modules
# =============================================================================

from typing import List

# Third-party
import jax
from jax.tree_util import tree_map
from jaxtyping import PyTree

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


def tree_flatten_population_dim(params: PyTree) -> PyTree:
    """
    Flatten the population dimension of a PyTree.

    Parameters
    ----------
    params : PyTree
        The PyTree with a population dimension to flatten.

    Returns
    -------
    PyTree
        The PyTree with the population dimension flattened.
    """
    def flatten_fn(x):
        # Get the shape of the parameter
        shape = x.shape
        # Assuming the shape is (batch_size, population_dim, *other_dims)
        batch_size, population_dim, *other_dims = shape
        # Reshape to (batch_size * population_dim, *other_dims)
        return x.reshape((batch_size * population_dim, *other_dims))

    # Apply flatten_fn to each element of the parameter tree
    return tree_map(flatten_fn, params)


def tree_to_dict_list(tree: PyTree, name: str) -> List[dict[str, PyTree]]:
    """
    Convert a PyTree into a list of dictionaries.

    Parameters
    ----------
    tree : PyTree
        The PyTree to convert.
    name : str
        The key name to use in the dictionaries.

    Returns
    -------
    List[Dict[str, PyTree]]
        A list of dictionaries representing the PyTree.
    """
    num_models = jax.tree_util.tree_leaves(
        tree)[0].shape[0]  # Get the batch size

    return [{name:
             jax.tree_util.tree_map(lambda param: param[i], tree)}
            for i in range(num_models)]


def dict_list_to_tree(dict_list: List[dict[str, PyTree]], name: str) -> PyTree:
    """
    Convert a list of dictionaries into a PyTree.

    Parameters
    ----------
    dict_list : List[Dict[str, PyTree]]
        The list of dictionaries to convert.
    name : str
        The key name to extract from the dictionaries.

    Returns
    -------
    PyTree
        The resulting PyTree.
    """
    return tree_map(lambda *params: jax.numpy.stack(params, axis=0),
                    *[d[name] for d in dict_list])
