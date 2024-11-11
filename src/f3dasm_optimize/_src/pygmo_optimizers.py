#                                                                       Modules
# =============================================================================

# Standard
from typing import List, Optional

# Third-party
import pygmo as pg

from ._protocol import Domain, OptimizerTuple
# Locals
from .adapters.pygmo_implementations import PygmoAlgorithm

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class CMAES(PygmoAlgorithm):
    """Covariance Matrix Adaptation Evolution Strategy optimizer
    implemented from pygmo"""
    require_gradients: bool = False

    def __init__(
        self, domain: Domain, population: int = 30,
            force_bounds: bool = True, seed: Optional[int] = None, **kwargs):
        super().__init__(domain=domain,
                         population=population, seed=seed)
        self.force_bounds = force_bounds
        self._set_algorithm()

    def _set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.cmaes(
                gen=1,
                memory=True,
                seed=self.seed,
                force_bounds=self.force_bounds,
            )
        )

    def _get_info(self) -> List[str]:
        return ['Stable', 'Global', 'Population-Based']


def cmaes_pygmo(population: int = 30,
                force_bounds: bool = True,
                seed: Optional[int] = None) -> OptimizerTuple:
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
    Adapted from pygmo.

    Parameters
    ----------
    population : int, optional
        The population size, by default 30
    force_bounds : bool, optional
        If bounds should be enforced, by default True
    seed : Optional[int], optional
        Seed for random number generation, by default None

    Returns
    -------
    OptimizerTuple
        A configured instance of the CMA-ES optimizer.
    """
    return OptimizerTuple(
        optimizer=CMAES,
        hyperparameters={
            'population': population,
            'force_bounds': force_bounds,
            'seed': seed
        })

# =============================================================================


class DifferentialEvolution(PygmoAlgorithm):
    "DifferentialEvolution optimizer implemented from pygmo"
    require_gradients: bool = False

    def __init__(self, domain: Domain, population: int = 30,
                 F: float = 0.8, CR: float = 0.9, variant: int = 2,
                 ftol: float = 0.0, xtol: float = 0.0,
                 seed: Optional[int] = None, **kwargs):
        super().__init__(domain=domain,
                         population=population, seed=seed)
        self.F = F
        self.CR = CR
        self.variant = variant
        self.ftol = ftol
        self.xtol = xtol
        self._set_algorithm()

    def _set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.de(
                gen=1,
                F=self.F,
                CR=self.CR,
                variant=self.variant,
                ftol=self.ftol,
                xtol=self.xtol,
                seed=self.seed,
            )
        )

    def _get_info(self) -> List[str]:
        return ['Fast', 'Global', 'Derivative-Free',
                'Population-Based', 'Single-Solution']


def de_pygmo(population: int = 30,
             F: float = 0.8,
             CR: float = 0.9,
             variant: int = 2,
             ftol: float = 0.0,
             xtol: float = 0.0,
             seed: Optional[int] = None) -> OptimizerTuple:
    """
    Differential Evolution optimizer using pygmo.

    Parameters
    ----------
    population : int, optional
        The population size, by default 30
    F : float, optional
        Scaling factor, by default 0.8
    CR : float, optional
        Crossover rate, by default 0.9
    variant : int, optional
        Variant to use, by default 2
    ftol : float, optional
        Function tolerance, by default 0.0
    xtol : float, optional
        Variable tolerance, by default 0.0
    seed : Optional[int], optional
        Seed for random number generation, by default None

    Returns
    -------
    PygmoAlgorithm
        A configured instance of the Differential Evolution optimizer.
    """
    return OptimizerTuple(
        optimizer=DifferentialEvolution,
        hyperparameters={
            'population': population,
            'F': F,
            'CR': CR,
            'variant': variant,
            'ftol': ftol,
            'xtol': xtol,
            'seed': seed
        })


# =============================================================================

class PygmoPSO(PygmoAlgorithm):
    """
    Particle Swarm Optimization (Generational) optimizer
    implemented from pygmo
    """
    require_gradients: bool = False

    def __init__(self, domain: Domain, population: int = 30,
                 eta1: float = 2.05, eta2: float = 2.05,
                 seed: Optional[int] = None, **kwargs):
        super().__init__(domain=domain,
                         population=population, seed=seed)
        self.eta1 = eta1
        self.eta2 = eta2
        self._set_algorithm()

    def _set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.pso_gen(
                gen=1,
                memory=True,
                seed=self.seed,
                eta1=self.eta1,
                eta2=self.eta2,
            )
        )

    def _get_info(self) -> List[str]:
        return ['Fast', 'Global', 'Derivative-Free',
                'Population-Based', 'Single-Solution']


def pso_pygmo(population: int = 30,
              eta1: float = 2.05,
              eta2: float = 2.05,
              seed: Optional[int] = None) -> OptimizerTuple:
    """
    Particle Swarm Optimization (Generational) using pygmo.

    Parameters
    ----------
    population : int, optional
        The population size, by default 30
    eta1 : float, optional
        Cognitive acceleration coefficient, by default 2.05
    eta2 : float, optional
        Social acceleration coefficient, by default 2.05
    seed : Optional[int], optional
        Seed for random number generation, by default None

    Returns
    -------
    PygmoAlgorithm
        A configured instance of the PSO optimizer.
    """
    return OptimizerTuple(
        optimizer=PygmoPSO,
        hyperparameters={
            'population': population,
            'eta1': eta1,
            'eta2': eta2,
            'seed': seed
        })


# =============================================================================


class SADE(PygmoAlgorithm):
    "Self-adaptive Differential Evolution optimizer implemented from pygmo"
    require_gradients: bool = False

    def __init__(self, domain: Domain, population: int = 30,
                 variant: int = 2, variant_adptv: int = 1,
                 ftol: float = 0.0, xtol: float = 0.0,
                 seed: Optional[int] = None, **kwargs):
        super().__init__(domain=domain,
                         population=population, seed=seed)
        self.variant = variant
        self.variant_adptv = variant_adptv
        self.ftol = ftol
        self.xtol = xtol
        self._set_algorithm()

    def _set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.sade(
                gen=1,
                variant=self.variant,
                variant_adptv=self.variant_adptv,
                ftol=self.ftol,
                xtol=self.xtol,
                memory=True,
                seed=self.seed,
            )
        )

    def _get_info(self) -> List[str]:
        return ['Fast', 'Population-Based', 'Single-Solution']


def sade(population: int = 30,
         variant: int = 2,
         variant_adptv: int = 1,
         ftol: float = 0.0,
         xtol: float = 0.0,
         seed: Optional[int] = None) -> OptimizerTuple:
    """
    Self-adaptive Differential Evolution (SADE) using pygmo.

    Parameters
    ----------
    population : int, optional
        The population size, by default 30
    variant : int, optional
        Variant to use, by default 2
    variant_adptv : int, optional
        Adaptive variant type, by default 1
    ftol : float, optional
        Function tolerance, by default 0.0
    xtol : float, optional
        Variable tolerance, by default 0.0
    seed : Optional[int], optional
        Seed for random number generation, by default None

    Returns
    -------
    PygmoAlgorithm
        A configured instance of the SADE optimizer.
    """
    return OptimizerTuple(
        optimizer=SADE,
        hyperparameters={
            'population': population,
            'variant': variant,
            'variant_adptv': variant_adptv,
            'ftol': ftol,
            'xtol': xtol,
            'seed': seed
        })


# =============================================================================

class SEA(PygmoAlgorithm):
    """Simple Evolutionary Algorithm optimizer implemented from pygmo"""
    require_gradients: bool = False

    def __init__(self, domain: Domain, population: int = 30,
                 seed: Optional[int] = None, **kwargs):
        super().__init__(domain=domain,
                         population=population, seed=seed)
        self._set_algorithm()

    def _set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.sea(
                gen=1,
                seed=self.seed,
            )
        )

    def _get_info(self) -> List[str]:
        return ['Fast', 'Global', 'Derivative-Free', 'Population-Based']


def sea(population: int = 30,
        seed: Optional[int] = None) -> OptimizerTuple:
    """
    Simple Evolutionary Algorithm (SEA) optimizer using pygmo.

    Parameters
    ----------
    population : int, optional
        The population size, by default 30
    seed : Optional[int], optional
        Seed for random number generation, by default None

    Returns
    -------
    PygmoAlgorithm
        A configured instance of the SEA optimizer.
    """
    return OptimizerTuple(
        optimizer=SEA,
        hyperparameters={
            'population': population,
            'seed': seed
        })


# =============================================================================


class SGA(PygmoAlgorithm):
    """Simple Genetic Algorithm optimizer implemented from pygmo"""
    require_gradients: bool = False

    def __init__(self, domain: Domain, population: int = 30,
                 cr: float = 0.9, eta_c: float = 1.0, m: float = 0.02,
                 param_m: float = 1.0, param_s: int = 2,
                 crossover: str = 'exponential', mutation: str = 'polynomial',
                 selection: str = 'tournament',
                 seed: Optional[int] = None, **kwargs):
        super().__init__(domain=domain,
                         population=population, seed=seed)
        self.cr = cr
        self.eta_c = eta_c
        self.m = m
        self.param_m = param_m
        self.param_s = param_s
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self._set_algorithm()

    def _set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.sga(
                gen=1,
                cr=self.cr,
                eta_c=self.eta_c,
                m=self.m,
                param_m=self.param_m,
                param_s=self.param_s,
                crossover=self.crossover,
                mutation=self.mutation,
                selection=self.selection,
                seed=self.seed,
            )
        )

    def _get_info(self) -> List[str]:
        return ['Fast', 'Population-Based']


def sga(population: int = 30,
        cr: float = 0.9,
        eta_c: float = 1.0,
        m: float = 0.02,
        param_m: float = 1.0,
        param_s: int = 2,
        crossover: str = 'exponential',
        mutation: str = 'polynomial',
        selection: str = 'tournament',
        seed: Optional[int] = None) -> OptimizerTuple:
    """
    Simple Genetic Algorithm (SGA) optimizer using pygmo.

    Parameters
    ----------
    population : int, optional
        The population size, by default 30
    cr : float, optional
        Crossover probability, by default 0.9
    eta_c : float, optional
        Crossover distribution index, by default 1.0
    m : float, optional
        Mutation probability, by default 0.02
    param_m : float, optional
        Mutation distribution index, by default 1.0
    param_s : int, optional
        Selection pressure, by default 2
    crossover : str, optional
        Crossover method, by default 'exponential'
    mutation : str, optional
        Mutation method, by default 'polynomial'
    selection : str, optional
        Selection method, by default 'tournament'
    seed : Optional[int], optional
        Seed for random number generation, by default None

    Returns
    -------
    PygmoAlgorithm
        A configured instance of the SGA optimizer.
    """
    return OptimizerTuple(
        optimizer=SGA,
        hyperparameters={
            'population': population,
            'cr': cr,
            'eta_c': eta_c,
            'm': m,
            'param_m': param_m,
            'param_s': param_s,
            'crossover': crossover,
            'mutation': mutation,
            'selection': selection,
            'seed': seed
        })


# =============================================================================


class SimulatedAnnealing(PygmoAlgorithm):
    "DifferentialEvolution optimizer implemented from pygmo"
    require_gradients: bool = False

    def __init__(self, domain: Domain, population: int = 30,
                 Ts: float = 10.0, Tf: float = 0.1, n_T_adj: int = 10,
                 n_range_adj: int = 10, bin_size: int = 10,
                 start_range: float = 1.0,
                 seed: Optional[int] = None, **kwargs):
        super().__init__(domain=domain,
                         population=population, seed=seed)
        self.Ts = Ts
        self.Tf = Tf
        self.n_T_adj = n_T_adj
        self.n_range_adj = n_range_adj
        self.bin_size = bin_size
        self.start_range = start_range
        self._set_algorithm()

    def _set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.simulated_annealing(
                Ts=self.Ts,
                Tf=self.Tf,
                n_T_adj=self.n_T_adj,
                n_range_adj=self.n_range_adj,
                bin_size=self.bin_size,
                start_range=self.start_range,
                seed=self.seed,
            )
        )

    def _get_info(self) -> List[str]:
        return ['Stable', 'Global', 'Derivative-Free', 'Single-Solution']


def simanneal_pygmo(population: int = 30,
                    Ts: float = 10.0,
                    Tf: float = 0.1,
                    n_T_adj: int = 10,
                    n_range_adj: int = 10,
                    bin_size: int = 10,
                    start_range: float = 1.0,
                    seed: Optional[int] = None) -> OptimizerTuple:
    """
    Simulated Annealing optimizer using pygmo.

    Parameters
    ----------
    population : int, optional
        The population size, by default 30
    Ts : float, optional
        Starting temperature, by default 10.0
    Tf : float, optional
        Final temperature, by default 0.1
    n_T_adj : int, optional
        Number of temperature adjustments, by default 10
    n_range_adj : int, optional
        Number of range adjustments, by default 10
    bin_size : int, optional
        Size of bins for temperature adjustment, by default 10
    start_range : float, optional
        Starting range for search, by default 1.0
    seed : Optional[int], optional
        Seed for random number generation, by default None

    Returns
    -------
    PygmoAlgorithm
        A configured instance of the Simulated Annealing optimizer.
    """
    return OptimizerTuple(
        optimizer=SimulatedAnnealing,
        hyperparameters={
            'population': population,
            'Ts': Ts,
            'Tf': Tf,
            'n_T_adj': n_T_adj,
            'n_range_adj': n_range_adj,
            'bin_size': bin_size,
            'start_range': start_range,
            'seed': seed
        })


# =============================================================================


class XNES(PygmoAlgorithm):
    """XNES optimizer implemented from pygmo"""
    require_gradients: bool = False

    def __init__(self, domain: Domain, population: int = 30,
                 eta_mu: float = -1.0, eta_sigma: float = -1.0,
                 eta_b: float = -1.0, sigma0: float = -1.0,
                 ftol: float = 1e-06, xtol: float = 1e-06,
                 force_bounds: bool = True,
                 seed: Optional[int] = None, **kwargs):
        super().__init__(domain=domain,
                         population=population, seed=seed)
        self.eta_mu = eta_mu
        self.eta_sigma = eta_sigma
        self.eta_b = eta_b
        self.sigma0 = sigma0
        self.ftol = ftol
        self.xtol = xtol
        self.force_bounds = force_bounds
        self._set_algorithm()

    def _set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.xnes(
                gen=1,
                eta_mu=self.eta_mu,
                eta_sigma=self.eta_sigma,
                eta_b=self.eta_b,
                sigma0=self.sigma0,
                ftol=self.ftol,
                xtol=self.xtol,
                memory=True,
                force_bounds=self.force_bounds,
                seed=self.seed,
            )
        )

    def _get_info(self) -> List[str]:
        return ['Stable', 'Global', 'Population-Based']


def xnes(population: int = 30,
         eta_mu: float = -1.0,
         eta_sigma: float = -1.0,
         eta_b: float = -1.0,
         sigma0: float = -1.0,
         ftol: float = 1e-06,
         xtol: float = 1e-06,
         force_bounds: bool = True,
         seed: Optional[int] = None) -> OptimizerTuple:
    """
    Exponential Natural Evolution Strategies (xNES) optimizer using pygmo.

    Parameters
    ----------
    population : int, optional
        The population size, by default 30
    eta_mu : float, optional
        Learning rate for the mean, by default -1.0
    eta_sigma : float, optional
        Learning rate for sigma, by default -1.0
    eta_b : float, optional
        Learning
    sigma0 : float, optional
        Initial sigma, by default -1.0
    ftol : float, optional
        Function tolerance, by default 1e-06
    xtol : float, optional
        Variable tolerance, by default 1e-06
    force_bounds : bool, optional
        If bounds should be enforced, by default True
    seed : Optional[int], optional
        Seed for random number generation, by default None

    Returns
    -------
    PygmoAlgorithm
        A configured instance of the xNES optimizer.
    """
    return OptimizerTuple(
        optimizer=XNES,
        hyperparameters={
            'population': population,
            'eta_mu': eta_mu,
            'eta_sigma': eta_sigma,
            'eta_b': eta_b,
            'sigma0': sigma0,
            'ftol': ftol,
            'xtol': xtol,
            'force_bounds': force_bounds,
            'seed': seed
        })
