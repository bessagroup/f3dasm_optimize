import pytest
from f3dasm import ExperimentData
from f3dasm.design import Domain


@pytest.fixture(scope="package")
def data():
    seed = 42
    N = 50  # Number of samples

    # Create the design space
    design = Domain()

    design.add_float(name='x1', low=2.4, high=10.3)
    design.add_float(name='x2', low=10.0, high=380.3)
    design.add_float(name='x3', low=0.6, high=7.3)

    # Set the lower_bound and upper_bound of 'y' to None, indicating it has no bounds
    return ExperimentData.from_sampling(sampler='random', domain=design, n_samples=N, seed=seed)
