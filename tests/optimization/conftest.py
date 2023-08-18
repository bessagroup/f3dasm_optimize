import pytest
from f3dasm.design.domain import Domain
from f3dasm.design.parameter import ContinuousParameter
from f3dasm.sampling.randomuniform import RandomUniform


@pytest.fixture(scope="package")
def data():
    seed = 42
    N = 50  # Number of samples

    # Define the parameters
    x1 = ContinuousParameter(lower_bound=2.4, upper_bound=10.3)
    x2 = ContinuousParameter(lower_bound=10.0, upper_bound=380.3)
    x3 = ContinuousParameter(lower_bound=0.6, upper_bound=7.3)
    # Create the design space
    input_space = {'x1': x1, 'x2': x2, 'x3': x3}
    design = Domain(input_space=input_space)

    random_sampler = RandomUniform(design=design, seed=seed)
    data = random_sampler.get_samples(numsamples=N)

    return data
