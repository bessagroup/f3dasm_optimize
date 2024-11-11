#                                                                       Modules
# =============================================================================

# Third-party
import nevergrad as ng

# Local
from ._protocol import Domain, OptimizerTuple
from .adapters.nevergrad_implementations import NeverGradOptimizer

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class NevergradDE(NeverGradOptimizer):
    require_gradients: bool = False

    def __init__(self, domain: Domain, population: int = 30,
                 initialization: str = 'parametrization', scale: float = 1.0,
                 recommendation: str = 'optimistic', crossover: float = 0.5,
                 F1: float = 0.8, F2: float = 0.8, **kwargs):

        super().__init__(domain=domain, population=population)
        self.initialization = initialization
        self.scale = scale
        self.recommendation = recommendation
        self.crossover = crossover
        self.F1 = F1
        self.F2 = F2
        self._set_algorithm()

    def _set_algorithm(self):
        p = ng.p.Array(shape=(len(self.domain),),
                       lower=self.domain.get_bounds()[:, 0],
                       upper=self.domain.get_bounds()[:, 1])
        self.algorithm = ng.optimizers.DifferentialEvolution(
            initialization=self.initialization,
            popsize=self.population,
            scale=self.scale,
            recommendation=self.recommendation,
            crossover=self.crossover,
            F1=self.F1,
            F2=self.F2)(p, budget=1e8)


def de_nevergrad(population: int = 30,
                 initialization: str = 'parametrization',
                 scale: float = 1.0,
                 recommendation: str = 'optimistic',
                 crossover: float = 0.5,
                 F1: float = 0.8,
                 F2: float = 0.8) -> NeverGradOptimizer:
    """
    Nevergrad Differential Evolution (DE) optimizer.

    Parameters
    ----------
    population : int, optional
        The number of individuals in the population, by default 30
    initialization : str, optional
        Initialization strategy, by default 'parametrization'
    scale : float, optional
        Scale factor, by default 1.0
    recommendation : str, optional
        Recommendation strategy, by default 'optimistic'
    crossover : float, optional
        Crossover probability, by default 0.5
    F1 : float, optional
        First differential weight, by default 0.8
    F2 : float, optional
        Second differential weight, by default 0.8

    Returns
    -------
    NeverGradOptimizer
        A configured instance of the Nevergrad DE optimizer.
    """
    return OptimizerTuple(
        optimizer=NevergradDE,
        hyperparameters={
            'population': population,
            'initialization': initialization,
            'scale': scale,
            'recommendation': recommendation,
            'crossover': crossover,
            'F1': F1,
            'F2': F2
        })

# =============================================================================


class PSO(NeverGradOptimizer):
    require_gradients: bool = False

    def __init__(
        self, domain: Domain, population: int = 30,
            transform: str = 'identity', omega: float = 0.7213475204444817,
            phip: float = 1.1931471805599454, phig: float = 1.1931471805599454,
            qo: bool = False, sqo: bool = False, so: bool = False, **kwargs):

        super().__init__(domain=domain, population=population)
        self.transform = transform
        self.omega = omega
        self.phip = phip
        self.phig = phig
        self.qo = qo
        self.sqo = sqo
        self.so = so
        self._set_algorithm()

    def _set_algorithm(self):
        p = ng.p.Array(shape=(len(self.domain),),
                       lower=self.domain.get_bounds()[:, 0],
                       upper=self.domain.get_bounds()[:, 1])
        self.algorithm = ng.optimizers.ConfPSO(
            transform=self.transform,
            popsize=self.population,
            omega=self.omega,
            phip=self.phip,
            phig=self.phig,
            qo=self.qo,
            sqo=self.sqo,
            so=self.so)(p, budget=1e8)


def pso_nevergrad(population: int = 30,
                  transform: str = 'identity',
                  omega: float = 0.7213475204444817,
                  phip: float = 1.1931471805599454,
                  phig: float = 1.1931471805599454,
                  qo: bool = False,
                  sqo: bool = False,
                  so: bool = False) -> NeverGradOptimizer:
    """
    Nevergrad Particle Swarm Optimization (PSO) optimizer.

    Parameters
    ----------
    population : int, optional
        The number of individuals in the population, by default 30
    transform : str, optional
        Transform strategy, by default 'identity'
    omega : float, optional
        Inertia weight, by default 0.7213475204444817
    phip : float, optional
        Personal attraction coefficient, by default 1.1931471805599454
    phig : float, optional
        Global attraction coefficient, by default 1.1931471805599454
    qo : bool, optional
        Use quasi-opposition, by default False
    sqo : bool, optional
        Use stochastic quasi-opposition, by default False
    so : bool, optional
        Use space opposition, by default False

    Returns
    -------
    NeverGradOptimizer
        A configured instance of the Nevergrad PSO optimizer.
    """
    return OptimizerTuple(
        optimizer=PSO,
        hyperparameters={
            'population': population,
            'transform': transform,
            'omega': omega,
            'phip': phip,
            'phig': phig,
            'qo': qo,
            'sqo': sqo,
            'so': so
        })
