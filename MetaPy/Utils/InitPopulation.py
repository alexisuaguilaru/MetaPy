import numpy as np

from typing import Callable

def RealValueIndividuals(
        LowerBound: float = -100,
        UpperBound: float = 100,
        Dimension: int = 2,
    ) -> Callable[[int],np.ndarray]:
    """
    Function for creating a population 
    initializer with real values 
    solutions/individuals.

    Parameters
    ----------
    LowerBound: float
        Lower bound for values of solutions
    UpperBound: float
        Upper bound for values of solutions
    Dimension: int
        Dimension of solutions/individuals. Number of values

    Return
    ------
    InitPopulation: Callable[[int],np.ndarray]
        Population initializer functions that takes a `PopulationSize` parameter and returns a `Population` of that size
    """
    
    def InitPopulation(
            PopulationSize: int,
        ) -> np.ndarray:
        """
        Function for generating a `Population` of 
        shape `(PopulationSize,Dim)` of real values.


        Parameters
        ----------
        PopulationSize: int
            Size of the `Population` to generate

        Return
        ------
        Population: np.ndarray
            Generated `Population` of shape `(PopulationSize,Dim)`
        """

        return np.random.uniform(LowerBound,UpperBound,size=(PopulationSize,Dimension))

    return InitPopulation